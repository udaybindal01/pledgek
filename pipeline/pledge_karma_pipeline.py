"""
PLEDGE-KARMA Full Pipeline

Orchestrates the complete end-to-end system:
  1. Knowledge state estimation (KARMA)
  2. Pedagogical retrieval (PLEDGE)
  3. Response generation with depth-aware prompting
  4. Knowledge state update from response
  5. Interaction logging

This is the entry point for both real-time inference and batch evaluation.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from models.mrl_encoder import MRLEncoder
from knowledge_graph.graph_builder import KnowledgeGraphBuilder
from karma.estimator import KARMAEstimator, Interaction
from pledge.retriever import PLEDGERetriever, RetrievalResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates (Depth-Aware)
# ─────────────────────────────────────────────────────────────────────────────

DEPTH_PROMPTS = {
    0: """You are a patient, encouraging tutor explaining concepts to a beginner.
Use simple language, relatable analogies, and avoid technical jargon.
Never assume background knowledge beyond basic high school math.
Connect the concept to everyday experience when possible.""",

    1: """You are a knowledgeable tutor working with an intermediate student.
You can use standard technical terminology but define any specialized terms.
The student has covered the prerequisite material but may need reminders.
Connect the explanation to what they already know when possible.""",

    2: """You are a rigorous academic tutor working with an advanced student.
Use formal mathematical notation where appropriate.
Assume solid background in the prerequisite material.
Connect to deeper theoretical implications and research context when relevant."""
}

REACTIVATION_PREFIX = """
⚠️ Note: Before answering the question, briefly reactivate the following
foundational concepts that are needed to understand the answer
(keep each to 1-2 sentences):
{concepts_to_reactivate}

Then answer the main question:
"""

SOCRATIC_TEMPLATE = """
Instead of directly answering, guide the student to discover the answer.
Ask 2-3 Socratic questions that lead them toward the insight.
Start with what they already know and build from there.
Main question to guide toward: {question}
Retrieved context for your reference (do not quote directly): {context}
"""

DIRECT_TEMPLATE = """
Using the following retrieved educational content, answer the student's question.
Depth level: {depth_description}

Retrieved Content:
{context}

Student Question: {question}

Provide a clear, pedagogically appropriate answer at the correct depth level.
If the question requires prerequisites the student hasn't covered,
acknowledge this and provide a bridge explanation first.
"""


@dataclass
class PipelineResponse:
    """Complete response from the PLEDGE-KARMA pipeline."""
    query: str
    answer: str
    response_mode: str                    # "direct", "socratic", "reactivation"
    retrieval_result: RetrievalResult
    depth_level_used: int
    reactivation_concepts: List[str]
    interaction_id: str
    timestamp: datetime
    metadata: Dict


class PLEDGEKARMAPipeline:
    """
    Complete PLEDGE-KARMA Pipeline.

    Usage:
        pipeline = PLEDGEKARMAPipeline(config, encoder, graph, karma, retriever, llm)
        response = pipeline.answer(query, student_id)
    """

    def __init__(
        self,
        config: Dict,
        encoder: MRLEncoder,
        graph: KnowledgeGraphBuilder,
        karma: KARMAEstimator,
        retriever: PLEDGERetriever,
        llm_client=None        # Anthropic/OpenAI client; None = mock
    ):
        self.config = config
        self.encoder = encoder
        self.graph = graph
        self.karma = karma
        self.retriever = retriever
        self.llm = llm_client

        self._interaction_counter = 0
        self._interaction_log: List[PipelineResponse] = []

        # Socratic routing thresholds
        karma_cfg = config.get("karma", {})
        mc_cfg = karma_cfg.get("metacognitive", {})
        self.socratic_readiness_threshold = 0.7  # Above this → try Socratic
        self.scaffold_threshold = 0.3            # Below this → direct scaffold

    def answer(
        self,
        query: str,
        student_concept_history: Optional[List[str]] = None,
        force_mode: Optional[str] = None  # Override routing for ablations
    ) -> PipelineResponse:
        """
        Main entry point. Full PLEDGE-KARMA pipeline.

        Args:
            query: Student's question
            student_concept_history: Recently visited concepts
            force_mode: "direct", "socratic", or None (auto-route)

        Returns:
            PipelineResponse with answer and all diagnostic metadata
        """
        self._interaction_counter += 1
        interaction_id = f"interaction_{self._interaction_counter}"
        timestamp = datetime.now()

        # ─── Phase 1: PLEDGE Retrieval ───
        logger.info(f"[{interaction_id}] Retrieving for query: {query[:80]}...")
        retrieval_result = self.retriever.retrieve(
            query=query,
            student_concept_history=student_concept_history
        )

        # ─── Phase 2: Response Mode Routing ───
        if force_mode:
            response_mode = force_mode
        else:
            response_mode = self._route_response_mode(
                retrieval_result, query
            )

        # ─── Phase 3: Check for reactivation needs ───
        reactivation_concepts = retrieval_result.reactivation_needed

        # ─── Phase 4: Determine depth level ───
        target_depth_float = retrieval_result.student_depth_level
        depth_level_int = int(round(target_depth_float))
        depth_level_int = max(0, min(2, depth_level_int))

        # ─── Phase 5: Build context string ───
        context = self._build_context_string(retrieval_result)

        # ─── Phase 6: Generate response ───
        answer = self._generate_response(
            query=query,
            context=context,
            response_mode=response_mode,
            depth_level=depth_level_int,
            reactivation_concepts=reactivation_concepts
        )

        # ─── Phase 7: Compute MRL divergence for KARMA update ───
        query_emb = self.encoder.encode_query(query)
        avg_mrl_divergence = 0.0
        if retrieval_result.retrieved_chunks:
            avg_mrl_divergence = float(
                sum(rc.mrl_divergence for rc in retrieval_result.retrieved_chunks)
                / len(retrieval_result.retrieved_chunks)
            )

        # ─── Phase 8: Update KARMA knowledge state ───
        # Estimate response quality from retrieval scores (proxy until human eval)
        proxy_quality = float(
            sum(rc.final_score for rc in retrieval_result.retrieved_chunks[:3])
            / max(len(retrieval_result.retrieved_chunks[:3]), 1)
        )

        touched_concepts = list({
            cid
            for rc in retrieval_result.retrieved_chunks
            for cid in rc.chunk.concept_ids
        })

        interaction = Interaction(
            interaction_id=interaction_id,
            timestamp=timestamp,
            query=query,
            concept_ids=touched_concepts,
            correct=None,             # No assessment; updated if quiz follows
            response_quality=proxy_quality,
            query_embedding_64d=(query_emb.at_dim(64) if query_emb else None),
            query_embedding_768d=(query_emb.at_dim(768) if query_emb else None),
            mrl_divergence=avg_mrl_divergence,
            skipped_concepts=[]
        )
        self.karma.update(interaction)

        # ─── Phase 9: Assemble response ───
        pipeline_response = PipelineResponse(
            query=query,
            answer=answer,
            response_mode=response_mode,
            retrieval_result=retrieval_result,
            depth_level_used=depth_level_int,
            reactivation_concepts=reactivation_concepts,
            interaction_id=interaction_id,
            timestamp=timestamp,
            metadata={
                "n_candidates_retrieved": len(retrieval_result.retrieved_chunks),
                "admissibility_violations": retrieval_result.admissibility_violations,
                "target_depth_float": target_depth_float,
                "avg_mrl_divergence": avg_mrl_divergence,
                "metacognitive_profile": retrieval_result.metacognitive_profile
            }
        )

        self._interaction_log.append(pipeline_response)
        return pipeline_response

    def update_from_assessment(
        self,
        interaction_id: str,
        concept_ids: List[str],
        correct: bool,
        response_quality: float = 0.8
    ) -> None:
        """
        Update KARMA with explicit assessment results (quiz, exercise).

        Called after a student completes an assessment following an interaction.
        This provides the hard correctness signal for BKT.
        """
        interaction = Interaction(
            interaction_id=f"{interaction_id}_assessment",
            timestamp=datetime.now(),
            query="[assessment]",
            concept_ids=concept_ids,
            correct=correct,
            response_quality=response_quality,
            mrl_divergence=0.0
        )
        self.karma.update(interaction)
        logger.info(
            f"Updated KARMA from assessment: {concept_ids} → "
            f"{'correct' if correct else 'incorrect'}"
        )

    def _route_response_mode(
        self,
        retrieval_result: RetrievalResult,
        query: str
    ) -> str:
        """
        Decide whether to answer directly or Socratically.

        Routing logic based on metacognitive profile and ZPD position:
        - Well-calibrated student at their frontier → Socratic
        - Overconfident student → Direct with gentle gap exposure
        - Underconfident student → Direct with scaffolding and encouragement
        - Missing prerequisites → Reactivation + Direct

        This implements the principled pedagogical routing from PLEDGE-KARMA theory.
        """
        if retrieval_result.reactivation_needed:
            return "reactivation"

        profile = retrieval_result.metacognitive_profile
        calibration = profile.get("calibration", "unknown")

        # Check prerequisite readiness from admissibility
        avg_admissibility = (
            sum(rc.admissibility_confidence for rc in retrieval_result.retrieved_chunks)
            / max(len(retrieval_result.retrieved_chunks), 1)
        )

        if avg_admissibility >= self.socratic_readiness_threshold:
            if calibration == "well-calibrated":
                return "socratic"
            elif calibration == "underconfident":
                # Underconfident + ready → Socratic to build confidence
                return "socratic"
            else:
                # Overconfident + high admissibility → direct with challenge
                return "direct"
        else:
            # Not ready → direct scaffolding
            return "direct"

    def _build_context_string(self, retrieval_result: RetrievalResult) -> str:
        """Format retrieved chunks into LLM context."""
        context_parts = []
        for i, rc in enumerate(retrieval_result.retrieved_chunks):
            context_parts.append(
                f"[Source {i+1} | Depth: {rc.chunk.depth_level} | "
                f"Admissibility: {rc.admissibility_confidence:.2f}]\n"
                f"{rc.chunk.text}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _generate_response(
        self,
        query: str,
        context: str,
        response_mode: str,
        depth_level: int,
        reactivation_concepts: List[str]
    ) -> str:
        """
        Generate the final response using depth-appropriate prompting.
        """
        system_prompt = DEPTH_PROMPTS.get(depth_level, DEPTH_PROMPTS[1])
        depth_descriptions = {
            0: "introductory — simple language, analogies, no jargon",
            1: "intermediate — standard terminology, assumes prerequisites",
            2: "advanced — formal notation, assumes strong background"
        }

        if response_mode == "reactivation" and reactivation_concepts:
            concepts_str = "\n".join(
                f"- {self.graph.concepts.get(cid, type('', (), {'name': cid})()).name}"
                for cid in reactivation_concepts[:3]
            )
            user_prompt = REACTIVATION_PREFIX.format(
                concepts_to_reactivate=concepts_str
            ) + DIRECT_TEMPLATE.format(
                depth_description=depth_descriptions.get(depth_level, ""),
                context=context[:3000],
                question=query
            )
        elif response_mode == "socratic":
            user_prompt = SOCRATIC_TEMPLATE.format(
                question=query,
                context=context[:2000]
            )
        else:
            user_prompt = DIRECT_TEMPLATE.format(
                depth_description=depth_descriptions.get(depth_level, ""),
                context=context[:3000],
                question=query
            )

        if self.llm is not None:
            return self._call_llm(system_prompt, user_prompt)
        else:
            # Mock response for testing
            return (
                f"[MOCK RESPONSE - Mode: {response_mode}, Depth: {depth_level}]\n"
                f"Query: {query}\n"
                f"Context sources: {len(context.split('---'))}\n"
                f"Reactivation needed: {reactivation_concepts}"
            )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the configured LLM (Anthropic Claude or OpenAI)."""
        try:
            # Try Anthropic
            import anthropic
            if isinstance(self.llm, anthropic.Anthropic):
                message = self.llm.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return message.content[0].text
        except Exception:
            pass

        try:
            # Try OpenAI
            import openai
            if hasattr(self.llm, 'chat'):
                response = self.llm.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1024
                )
                return response.choices[0].message.content
        except Exception:
            pass

        return f"[LLM call failed] Prompt: {user_prompt[:200]}..."

    def get_interaction_log(self) -> List[Dict]:
        """Return serializable interaction log for analysis."""
        return [
            {
                "interaction_id": r.interaction_id,
                "timestamp": r.timestamp.isoformat(),
                "query": r.query,
                "response_mode": r.response_mode,
                "depth_level": r.depth_level_used,
                "n_chunks_retrieved": len(r.retrieval_result.retrieved_chunks),
                "admissibility_violations": r.retrieval_result.admissibility_violations,
                "reactivation_needed": r.reactivation_concepts,
                "metacognitive_profile": r.metadata.get("metacognitive_profile", {}),
                "avg_mrl_divergence": r.metadata.get("avg_mrl_divergence", 0.0)
            }
            for r in self._interaction_log
        ]

    def save_state(self, path: str) -> None:
        """Save complete pipeline state."""
        import os
        os.makedirs(path, exist_ok=True)
        self.karma.save(f"{path}/karma_state.json")
        with open(f"{path}/interaction_log.json", "w") as f:
            json.dump(self.get_interaction_log(), f, indent=2)
        logger.info(f"Pipeline state saved to {path}")