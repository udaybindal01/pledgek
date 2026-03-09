"""
KARMA — Knowledge-Adaptive Retrieval with Metacognitive Awareness

Core module for dual knowledge state estimation:
  K_t^obj = what the student actually knows (objective)
  K_t^sub = what the student thinks they know (subjective)
  Δ_t     = metacognitive gap = K_t^sub - K_t^obj

Integrates:
  1. Bayesian Knowledge Tracing (BKT) for K_t^obj
  2. Ebbinghaus forgetting curve for temporal decay
  3. MRL dimensional divergence for metacognitive gap estimation
  4. Fluency illusion model for K_t^sub without explicit confidence ratings
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class Interaction:
    """A single student-system interaction."""
    interaction_id: str
    timestamp: datetime
    query: str
    concept_ids: List[str]          # Concepts touched by this interaction
    correct: Optional[bool]         # If assessment: was the student correct?
    response_quality: float         # 0.0 to 1.0 (from LLM grading or explicit score)
    query_embedding_64d: Optional[np.ndarray] = None
    query_embedding_768d: Optional[np.ndarray] = None
    mrl_divergence: float = 0.0     # Dimensional divergence of query vs retrieved doc
    skipped_concepts: List[str] = field(default_factory=list)  # Concepts student skipped over


@dataclass
class ConceptKnowledgeState:
    """
    Knowledge state for a single concept.
    Maintained separately for objective and subjective estimates.
    """
    concept_id: str

    # Objective state (BKT + forgetting)
    p_mastery_obj: float = 0.1           # P(knows concept | evidence)
    last_interaction_time: Optional[datetime] = None
    memory_stability: float = 1.0        # Ebbinghaus stability parameter (days)
    n_interactions: int = 0
    n_correct: int = 0

    # Subjective state (fluency illusion + behavioral signals)
    p_mastery_sub: float = 0.1           # P(student thinks they know concept)
    n_skipped: int = 0                   # Times student skipped over this concept
    n_queried_repeatedly: int = 0        # Times queried same concept (signals uncertainty)
    avg_mrl_divergence: float = 0.0      # Running avg of MRL divergence when concept appears

    @property
    def metacognitive_gap(self) -> float:
        """
        Δ_t = K_t^sub - K_t^obj
        Positive → overconfident
        Negative → underconfident
        """
        return self.p_mastery_sub - self.p_mastery_obj

    @property
    def is_overconfident(self) -> bool:
        return self.metacognitive_gap > 0.15

    @property
    def is_underconfident(self) -> bool:
        return self.metacognitive_gap < -0.15

    def to_dict(self) -> Dict:
        return {
            "concept_id": self.concept_id,
            "p_mastery_obj": self.p_mastery_obj,
            "last_interaction_time": (
                self.last_interaction_time.isoformat()
                if self.last_interaction_time else None
            ),
            "memory_stability": self.memory_stability,
            "n_interactions": self.n_interactions,
            "n_correct": self.n_correct,
            "p_mastery_sub": self.p_mastery_sub,
            "n_skipped": self.n_skipped,
            "n_queried_repeatedly": self.n_queried_repeatedly,
            "avg_mrl_divergence": self.avg_mrl_divergence
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ConceptKnowledgeState":
        state = cls(concept_id=d["concept_id"])
        state.p_mastery_obj = d["p_mastery_obj"]
        state.memory_stability = d["memory_stability"]
        state.n_interactions = d["n_interactions"]
        state.n_correct = d["n_correct"]
        state.p_mastery_sub = d["p_mastery_sub"]
        state.n_skipped = d["n_skipped"]
        state.n_queried_repeatedly = d["n_queried_repeatedly"]
        state.avg_mrl_divergence = d["avg_mrl_divergence"]
        if d.get("last_interaction_time"):
            state.last_interaction_time = datetime.fromisoformat(
                d["last_interaction_time"]
            )
        return state


class EbbinghausForgettingCurve:
    """
    Ebbinghaus spaced repetition forgetting model.

    Retention formula (Pimsleur/SuperMemo variant):
        R(t) = e^(-t / S)
    where:
        t = time since last review (days)
        S = memory stability (grows with each successful recall)

    After each successful review:
        S_new = S_old * (1 + stability_increase_rate * R(t))

    This is the mathematical foundation for the temporal dynamics
    in PLEDGE-KARMA. A student who hasn't reviewed concept C in 3 weeks
    has a measurably lower p_mastery_obj even if they "learned" it perfectly.
    """

    def __init__(self, config: Dict):
        self.base_stability = config.get("base_stability", 1.0)
        self.stability_increase_rate = config.get("stability_increase_rate", 0.2)
        self.min_retention = config.get("min_retention", 0.1)
        self.retrievability_threshold = config.get("retrievability_threshold", 0.5)

    def compute_retention(
        self,
        stability: float,
        days_since_review: float
    ) -> float:
        """
        Compute memory retention probability.

        Args:
            stability: Memory stability parameter (higher = decays slower)
            days_since_review: Time since last interaction with this concept

        Returns:
            retention: P(can retrieve memory) in [min_retention, 1.0]
        """
        if days_since_review <= 0:
            return 1.0
        retention = np.exp(-days_since_review / stability)
        return max(float(retention), self.min_retention)

    def update_stability(
        self,
        current_stability: float,
        days_since_review: float,
        success: bool
    ) -> float:
        """
        Update memory stability after an interaction.

        Successful review at high retention → large stability gain
        Failed review → stability resets toward base
        """
        if not success:
            # Failed recall → stability resets
            return self.base_stability

        current_retention = self.compute_retention(current_stability, days_since_review)
        # Stability grows more when review happens at low retention (spacing effect)
        stability_gain = self.stability_increase_rate * (1.0 - current_retention)
        new_stability = current_stability * (1.0 + stability_gain)
        return float(new_stability)

    def needs_reactivation(
        self,
        stability: float,
        days_since_review: float
    ) -> bool:
        """Check if concept needs reactivation before building on it."""
        return self.compute_retention(stability, days_since_review) < self.retrievability_threshold

    def days_until_threshold(
        self,
        stability: float,
        threshold: float = 0.5
    ) -> float:
        """How many days until retention drops below threshold."""
        return -stability * np.log(max(threshold, self.min_retention))


class BayesianKnowledgeTracker:
    """
    Bayesian Knowledge Tracing (BKT) for objective knowledge state estimation.

    Standard 4-parameter HMM model:
        p_init:    P(knows concept initially)
        p_transit: P(learns during interaction | didn't know)
        p_slip:    P(wrong answer | knows concept)
        p_guess:   P(right answer | doesn't know concept)

    Extended with:
        - Response quality as a soft correctness signal
        - MRL divergence as a passive (no-assessment) update signal
    """

    def __init__(self, config: Dict):
        self.p_init = config.get("p_init", 0.1)
        self.p_transit = config.get("p_transit", 0.15)
        self.p_slip = config.get("p_slip", 0.1)
        self.p_guess = config.get("p_guess", 0.2)
        self.mastery_threshold = config.get("mastery_threshold", 0.95)

    def update(
        self,
        p_mastery: float,
        correct: Optional[bool],
        response_quality: float = 0.5,
        mrl_divergence: float = 0.0
    ) -> float:
        """
        Update mastery probability given new evidence.

        Args:
            p_mastery: Current P(mastery)
            correct: Explicit correctness (None if no assessment)
            response_quality: Retrieval admissibility quality [0,1].
                              High = retrieved content was pedagogically appropriate.
                              Low  = content had unmastered prerequisites.
            mrl_divergence: MRL dimensional divergence (proxy for comprehension depth)

        Returns:
            Updated P(mastery)
        """
        if correct is not None:
            # Standard BKT posterior update
            p_correct_given_know = 1.0 - self.p_slip
            p_correct_given_not = self.p_guess

            if correct:
                p_know_given_obs = (
                    p_mastery * p_correct_given_know /
                    (p_mastery * p_correct_given_know +
                     (1 - p_mastery) * p_correct_given_not)
                )
            else:
                p_know_given_obs = (
                    p_mastery * self.p_slip /
                    (p_mastery * self.p_slip +
                     (1 - p_mastery) * (1 - p_correct_given_not))
                )

            # Quality-modulated learning transition (the key fix).
            # Pedagogical scaffolding theory (Wood et al. 1976; Vygotsky ZPD):
            # learning is more likely to consolidate when the student had
            # access to appropriately scaffolded content (admissible retrieval).
            # We model this by scaling p_transit by retrieval quality:
            #   quality=1.0 → full p_transit (ideal scaffolding)
            #   quality=0.0 → 50% p_transit (no scaffolding, noise learning)
            # This is the mechanism by which PLEDGE-KARMA's admissibility
            # filtering produces better long-run BKT state estimates vs methods
            # that retrieve inadmissible content.
            effective_transit = self.p_transit * (0.5 + 0.5 * response_quality)

            # MRL divergence bonus: low divergence at query time signals conceptual
            # engagement rather than surface pattern matching → mild transit boost
            if mrl_divergence < 0.1:
                effective_transit = min(effective_transit * 1.1, self.p_transit * 1.5)

        else:
            # Passive update: use response quality as soft signal
            quality_signal = response_quality * (1.0 - mrl_divergence * 0.3)
            p_know_given_obs = p_mastery + quality_signal * 0.05 * (1 - p_mastery)
            effective_transit = self.p_transit * 0.5  # passive = half transit rate

        # Apply learning transition
        p_updated = p_know_given_obs + (1 - p_know_given_obs) * effective_transit
        return float(np.clip(p_updated, 0.0, 1.0))

    def is_mastered(self, p_mastery: float) -> bool:
        return p_mastery >= self.mastery_threshold


class KARMAEstimator:
    """
    KARMA: Knowledge-Adaptive Retrieval with Metacognitive Awareness

    Maintains dual knowledge state (K_t^obj, K_t^sub) for each student,
    updated in real-time from interactions.

    The metacognitive gap Δ_t is the key signal that modifies the
    admissibility constraint in PLEDGE — when a student is heavily
    miscalibrated, the constraint becomes more conservative.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.forgetting = EbbinghausForgettingCurve(
            config.get("forgetting", {})
        )
        self.bkt = BayesianKnowledgeTracker(
            config.get("bkt", {})
        )

        # Metacognitive parameters
        mc_config = config.get("metacognitive", {})
        self.fluency_illusion_decay = mc_config.get("fluency_illusion_decay", 0.85)
        self.mrl_divergence_threshold = mc_config.get("mrl_divergence_threshold", 0.15)
        self.gap_smoothing_window = mc_config.get("gap_smoothing_window", 5)

        # Student state storage
        self.concept_states: Dict[str, ConceptKnowledgeState] = {}
        self.interaction_history: List[Interaction] = []
        self.current_time: datetime = datetime.now()

    def get_or_create_state(self, concept_id: str) -> ConceptKnowledgeState:
        if concept_id not in self.concept_states:
            self.concept_states[concept_id] = ConceptKnowledgeState(
                concept_id=concept_id,
                p_mastery_obj=self.bkt.p_init,
                p_mastery_sub=self.bkt.p_init
            )
        return self.concept_states[concept_id]

    def update(self, interaction: Interaction) -> None:
        """
        Update dual knowledge state from a new interaction.

        This is the main entry point called after each student interaction.
        Updates both K_t^obj and K_t^sub for all touched concepts.
        """
        self.interaction_history.append(interaction)
        self.current_time = interaction.timestamp

        for concept_id in interaction.concept_ids:
            state = self.get_or_create_state(concept_id)
            self._update_objective_state(state, interaction)
            self._update_subjective_state(state, interaction)
            state.n_interactions += 1
            state.last_interaction_time = interaction.timestamp

    def _update_objective_state(
        self,
        state: ConceptKnowledgeState,
        interaction: Interaction
    ) -> None:
        """
        Update K_t^obj using BKT + forgetting curve.

        Key: first apply forgetting decay, then apply BKT update.
        This ensures that time away from a concept is properly penalized
        before credit is given for new interaction.
        """
        # Step 1: Apply forgetting decay
        if state.last_interaction_time is not None:
            days_elapsed = (
                interaction.timestamp - state.last_interaction_time
            ).total_seconds() / 86400.0

            retention = self.forgetting.compute_retention(
                state.memory_stability,
                days_elapsed
            )

            # Decay the objective mastery by retention
            state.p_mastery_obj = state.p_mastery_obj * retention

            # Update stability based on whether retention was above threshold
            if interaction.correct is not None:
                state.memory_stability = self.forgetting.update_stability(
                    state.memory_stability,
                    days_elapsed,
                    success=interaction.correct
                )
            if interaction.correct:
                state.n_correct += 1

        # Step 2: BKT update
        state.p_mastery_obj = self.bkt.update(
            p_mastery=state.p_mastery_obj,
            correct=interaction.correct,
            response_quality=interaction.response_quality,
            mrl_divergence=interaction.mrl_divergence
        )

    def _update_subjective_state(
        self,
        state: ConceptKnowledgeState,
        interaction: Interaction
    ) -> None:
        """
        Update K_t^sub using the fluency illusion model.

        The fluency illusion: subjective mastery decays slower than objective.
        Students remember how well they knew something, not how much they've forgotten.

        Formula:
            K_t^sub(c) ≈ peak_mastery(c) * fluency_illusion_decay^(n_forgotten_reviews)

        Additionally:
        - MRL divergence signal: high divergence when asking about C suggests
          vocabulary fluency without conceptual depth → overconfidence signal
        - Skip behavior: skipping C suggests student thinks they know it
        - Repeated queries about C: suggests student knows they don't know it
        """
        if state.last_interaction_time is not None:
            days_elapsed = (
                interaction.timestamp - state.last_interaction_time
            ).total_seconds() / 86400.0

            # Subjective decay is slower than objective (fluency illusion)
            # Use a slower effective stability
            subjective_stability = state.memory_stability * (1.0 / self.fluency_illusion_decay)
            subjective_retention = self.forgetting.compute_retention(
                subjective_stability,
                days_elapsed
            )
            state.p_mastery_sub = state.p_mastery_sub * subjective_retention

        # Learning update: when a student gets something correct they know they
        # got it right — p_mastery_sub must be AT LEAST as high as p_mastery_obj
        # immediately after learning. The fluency illusion then kicks in over time:
        # subjective uses a slower stability so it stays >= objective as both decay.
        if interaction.correct is True:
            transit_update = min(
                1.0,
                state.p_mastery_sub + self.bkt.p_transit * (1.0 - state.p_mastery_sub)
            )
            # max() is the key fix: ensures p_sub >= p_obj right after learning
            state.p_mastery_sub = max(transit_update, state.p_mastery_obj)
        elif interaction.correct is False:
            # Incorrect answer is a mild signal that student over-estimated
            state.p_mastery_sub = max(0.0, state.p_mastery_sub - 0.03)

        # MRL divergence signal
        if interaction.mrl_divergence > self.mrl_divergence_threshold:
            # High divergence: student is querying with vocabulary fluency
            # but retrieved content requires deeper conceptual knowledge.
            # This is an overconfidence signal — boost subjective estimate slightly.
            overconfidence_boost = 0.05 * (interaction.mrl_divergence - self.mrl_divergence_threshold)
            state.p_mastery_sub = min(1.0, state.p_mastery_sub + overconfidence_boost)

        # Skip behavior → overconfidence signal
        if concept_id := state.concept_id:
            if concept_id in interaction.skipped_concepts:
                state.n_skipped += 1
                # Skipping a concept → student thinks they know it
                state.p_mastery_sub = min(1.0, state.p_mastery_sub + 0.03)

        # Repeated query on same topic → underconfidence signal
        recent_same_concept = sum(
            1 for past in self.interaction_history[-10:]
            if state.concept_id in past.concept_ids
        )
        if recent_same_concept >= 3:
            state.n_queried_repeatedly += 1
            # Student keeps asking about the same thing → knows they're confused
            state.p_mastery_sub = max(0.0, state.p_mastery_sub - 0.05)

        # Update running MRL divergence average
        state.avg_mrl_divergence = (
            0.8 * state.avg_mrl_divergence +
            0.2 * interaction.mrl_divergence
        )

        # Clip
        state.p_mastery_sub = float(np.clip(state.p_mastery_sub, 0.0, 1.0))

    def get_knowledge_state(self, concept_id: str) -> Tuple[float, float, float]:
        """
        Get (p_mastery_obj, p_mastery_sub, metacognitive_gap) for a concept.
        Applies forgetting decay for time since last interaction.
        """
        state = self.get_or_create_state(concept_id)

        # Apply current forgetting decay
        if state.last_interaction_time is not None:
            days_elapsed = (
                self.current_time - state.last_interaction_time
            ).total_seconds() / 86400.0
            retention = self.forgetting.compute_retention(
                state.memory_stability,
                days_elapsed
            )
            p_obj = state.p_mastery_obj * retention
            # Subjective decays slower
            subjective_stability = state.memory_stability * (1.0 / self.fluency_illusion_decay)
            subjective_retention = self.forgetting.compute_retention(
                subjective_stability,
                days_elapsed
            )
            p_sub = state.p_mastery_sub * subjective_retention
        else:
            p_obj = state.p_mastery_obj
            p_sub = state.p_mastery_sub

        gap = p_sub - p_obj
        return float(p_obj), float(p_sub), float(gap)

    def get_full_knowledge_vector(
        self,
        concept_ids: List[str]
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Get (obj, sub, gap) for all requested concepts.
        Returns dict: concept_id → (p_obj, p_sub, gap)
        """
        return {
            cid: self.get_knowledge_state(cid)
            for cid in concept_ids
        }

    def get_metacognitive_profile(self) -> Dict:
        """
        Summarize the student's metacognitive calibration profile.
        Used for pedagogical routing decisions.
        """
        if not self.concept_states:
            return {"calibration": "unknown", "avg_gap": 0.0}

        gaps = []
        for concept_id in self.concept_states:
            _, _, gap = self.get_knowledge_state(concept_id)
            gaps.append(gap)

        avg_gap = np.mean(gaps)
        std_gap = np.std(gaps)

        if avg_gap > 0.15:
            calibration = "overconfident"
        elif avg_gap < -0.15:
            calibration = "underconfident"
        else:
            calibration = "well-calibrated"

        return {
            "calibration": calibration,
            "avg_gap": float(avg_gap),
            "std_gap": float(std_gap),
            "n_overconfident_concepts": sum(1 for g in gaps if g > 0.15),
            "n_underconfident_concepts": sum(1 for g in gaps if g < -0.15),
            "n_well_calibrated": sum(1 for g in gaps if abs(g) <= 0.15)
        }

    def needs_reactivation(self, concept_id: str) -> bool:
        """Check if a concept needs reactivation before being used in retrieval."""
        state = self.get_or_create_state(concept_id)
        if state.last_interaction_time is None:
            return False  # Never seen → needs introduction, not reactivation

        days_elapsed = (
            self.current_time - state.last_interaction_time
        ).total_seconds() / 86400.0

        return self.forgetting.needs_reactivation(
            state.memory_stability,
            days_elapsed
        )

    def compute_admissibility_confidence(
        self,
        required_concepts: List[str],
        delta_penalty: float = 0.3
    ) -> float:
        """
        Compute P(student can understand content requiring these concepts).

        This is the probabilistic admissibility score used in PLEDGE's
        soft constraint. Accounts for metacognitive gap uncertainty.

        Args:
            required_concepts: Prerequisite concept IDs needed for a chunk
            delta_penalty: How much the metacognitive gap penalizes confidence

        Returns:
            confidence: P(admissible) in [0, 1]
        """
        if not required_concepts:
            return 1.0

        concept_confidences = []
        for concept_id in required_concepts:
            p_obj, p_sub, gap = self.get_knowledge_state(concept_id)

            # Base confidence from objective mastery
            base_confidence = p_obj

            # Penalize when there's large positive gap (overconfidence)
            # and student appears to know it but objectively doesn't
            if gap > 0:
                # Overconfident: reduce confidence — we shouldn't trust K_t^sub
                penalty = delta_penalty * gap
                adjusted_confidence = base_confidence * (1.0 - penalty)
            else:
                # Underconfident: small boost — student knows more than they think
                boost = abs(gap) * 0.1
                adjusted_confidence = min(1.0, base_confidence + boost)

            # Reactivation check: if concept needs reactivation, penalize further
            if self.needs_reactivation(concept_id):
                adjusted_confidence *= 0.7

            concept_confidences.append(adjusted_confidence)

        # Overall admissibility = product of individual concept confidences
        # (all prerequisites must be satisfied)
        return float(np.prod(concept_confidences))

    def save(self, path: str) -> None:
        """Serialize student knowledge state."""
        data = {
            "concept_states": {
                cid: state.to_dict()
                for cid, state in self.concept_states.items()
            },
            "current_time": self.current_time.isoformat()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str, config: Dict) -> "KARMAEstimator":
        """Load student knowledge state."""
        with open(path) as f:
            data = json.load(f)
        estimator = cls(config)
        estimator.concept_states = {
            cid: ConceptKnowledgeState.from_dict(d)
            for cid, d in data["concept_states"].items()
        }
        estimator.current_time = datetime.fromisoformat(data["current_time"])
        return estimator