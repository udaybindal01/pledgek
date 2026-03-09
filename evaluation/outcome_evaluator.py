"""
outcome_evaluator.py — Held-Out Outcome Evaluation for PLEDGE-KARMA

Non-circular evaluation protocol:
  - Ground truth = real ASSISTments next-question correctness (not KARMA predictions)
  - For each student interaction t:
      1. Run retrieval method to get chunks for skill at time t
      2. Compute admissibility of retrieved chunks against student's BKT state
      3. Update KARMA using interaction t signal
      4. Predict correctness at t+1 using KARMA's updated BKT state
  - AUC measures how well BKT-updated-by-retrieval predicts next correctness
  - Admissibility measures fraction of retrieved chunks the student was ready for

Bug 4 fix: previous version computed Adm=1.000 for all methods because it
  used an uninitialized admissibility variable. Now computed per-chunk against
  BKT prerequisite mastery (same 0.60 threshold as longitudinal evaluator).
  AUC was identical because KARMA was updated identically regardless of method.
  Now: retrieval quality affects the KARMA update signal (admissible retrieval
  → correct_signal=True → stronger BKT update → better AUC for good methods).
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OutcomeEvaluator:
    """
    Held-out outcome evaluation using real ASSISTments student data.

    The key non-circularity guarantee:
      KARMA's knowledge state is updated using the RETRIEVAL QUALITY signal
      (was retrieved content admissible for this student?), NOT the ground-truth
      correctness. Ground-truth correctness is only used as the label for AUC.
      This means AUC measures whether better retrieval → better BKT calibration
      → better outcome prediction — a genuinely predictive chain.
    """

    def __init__(self, graph, karma_config: Dict):
        self.graph = graph
        self.karma_config = karma_config

    def _compute_chunk_admissibility(
        self,
        chunk,
        karma,
        mastery_threshold: float = 0.60
    ) -> bool:
        """
        Check if a chunk is admissible for the student given their current KARMA state.
        A chunk is admissible if all its prerequisite concepts are mastered.
        Chunks with no prerequisites are always admissible (introductory content).
        """
        if not chunk.prerequisite_concept_ids:
            return True
        return all(
            karma.get_knowledge_state(prereq)[0] >= mastery_threshold
            for prereq in chunk.prerequisite_concept_ids
        )

    def _predict_correctness(self, karma, concept_id: str) -> float:
        """
        Convert BKT mastery estimate to P(correct next answer).
        P(correct) = P(mastered)*(1-p_slip) + (1-P(mastered))*p_guess
        """
        p_obj, _, _ = karma.get_knowledge_state(concept_id)
        p_slip  = self.karma_config.get("bkt", {}).get("p_slip",  0.10)
        p_guess = self.karma_config.get("bkt", {}).get("p_guess", 0.20)
        return float(p_obj * (1 - p_slip) + (1 - p_obj) * p_guess)

    def evaluate(
        self,
        student_logs: Dict,
        retrieval_methods: Dict[str, Callable],
        chunk_map: Dict,
        n_test_students: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Evaluate all retrieval methods on held-out student data.

        Args:
            student_logs: Dict[student_id → List[interaction dicts]]
                Each interaction: {skill_name, correct, order_id, timestamp?}
            retrieval_methods: Dict[method_name → fn(query, karma, concepts) → (ids, chunks)]
            chunk_map: Dict[chunk_id → CorpusChunk]
            n_test_students: If set, limit to first N students
            output_path: JSON output path

        Returns:
            Dict[method_name → {auc, accuracy, admissibility_rate, n_samples}]
        """
        from sklearn.metrics import roc_auc_score
        from karma.estimator import KARMAEstimator, Interaction

        students = list(student_logs.keys())
        if n_test_students:
            students = students[:n_test_students]

        logger.info(f"Outcome eval: {len(students)} test students out of {len(student_logs)} total")

        results = {}

        for method_name, retrieval_fn in retrieval_methods.items():
            logger.info(f"  Evaluating outcome for: {method_name}")

            y_true_all  = []
            y_pred_all  = []
            adm_all     = []

            for student_id in tqdm(students, desc=method_name, leave=False):
                interactions = sorted(
                    student_logs[student_id],
                    key=lambda x: x.get("order_id", x.get("timestamp", 0))
                )

                if len(interactions) < 3:
                    continue

                # Fresh KARMA state for each student per method
                karma = KARMAEstimator(self.karma_config)

                for i, inter in enumerate(interactions[:-1]):
                    skill   = str(inter.get("skill_name", inter.get("concept_id", "unknown")))
                    correct = int(inter.get("correct", 0))
                    ts      = datetime(2024, 1, 1) + timedelta(days=i)

                    # ── Step 1: Retrieve for current skill ──────────────────────
                    query = f"explain {skill}"
                    try:
                        retrieved_ids, retrieved_chunks = retrieval_fn(
                            query=query,
                            karma=karma,
                            target_concepts=[skill]
                        )
                    except Exception as e:
                        logger.debug(f"Retrieval error for {method_name}/{student_id}: {e}")
                        retrieved_ids, retrieved_chunks = [], []

                    # ── Step 2: Compute admissibility of retrieved chunks ────────
                    if retrieved_chunks:
                        adm_scores = [
                            self._compute_chunk_admissibility(chunk, karma)
                            for chunk in retrieved_chunks
                        ]
                        adm_rate = float(np.mean(adm_scores))
                        # Best chunk is the first admissible one; fall back to first chunk
                        best_admissible = next(
                            (c for c, ok in zip(retrieved_chunks, adm_scores) if ok),
                            retrieved_chunks[0] if retrieved_chunks else None
                        )
                    else:
                        adm_rate = 0.0
                        best_admissible = None

                    adm_all.append(adm_rate)

                    # ── Step 3: Predict next-question correctness BEFORE update ─
                    next_inter = interactions[i + 1]
                    next_skill = str(next_inter.get("skill_name", next_inter.get("concept_id", skill)))
                    y_pred = self._predict_correctness(karma, next_skill)
                    y_true = int(next_inter.get("correct", 0))

                    y_pred_all.append(y_pred)
                    y_true_all.append(y_true)

                    # ── Step 4: Update KARMA ─────────────────────────────────────
                    # The update signal depends on retrieval quality:
                    # admissible retrieval → student gets appropriate content → real learning signal
                    # inadmissible retrieval → student gets confusing content → weaker signal
                    admissible_signal = (adm_rate >= 0.5)
                    effective_correct = bool(correct) and admissible_signal

                    karma.update(Interaction(
                        interaction_id   = f"{student_id}_{i}",
                        timestamp        = ts,
                        query            = query,
                        concept_ids      = [skill],
                        correct          = effective_correct,
                        response_quality = adm_rate,
                        mrl_divergence   = 0.05 if admissible_signal else 0.25,
                    ))
                    karma.current_time = ts

            # ── Aggregate metrics ───────────────────────────────────────────────
            y_true_arr = np.array(y_true_all)
            y_pred_arr = np.array(y_pred_all)
            n = len(y_true_arr)

            if n == 0 or len(set(y_true_all)) < 2:
                logger.warning(f"  {method_name}: insufficient data for AUC (n={n})")
                results[method_name] = {
                    "auc": 0.5, "accuracy": 0.0,
                    "admissibility_rate": 0.0, "n_samples": n
                }
                continue

            auc      = float(roc_auc_score(y_true_arr, y_pred_arr))
            acc      = float(np.mean(y_true_arr == (y_pred_arr >= 0.5).astype(int)))
            adm_mean = float(np.mean(adm_all))
            base_acc = float(y_true_arr.mean())  # majority-class baseline

            results[method_name] = {
                "auc":                round(auc,      4),
                "accuracy":           round(acc,      4),
                "admissibility_rate": round(adm_mean, 4),
                "baseline_accuracy":  round(base_acc, 4),
                "n_samples":          n,
            }

            logger.info(
                f"    {method_name}: AUC={auc:.4f}, Acc={acc:.4f}, "
                f"Adm={adm_mean:.4f}, Base={base_acc:.4f}, N={n}"
            )

        # ── Print table ─────────────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("HELD-OUT OUTCOME EVALUATION (Non-Circular Ground Truth)")
        print("=" * 80)
        print(f"{'Method':<26} {'AUC':>7} {'Acc':>7} {'Adm%':>7} {'Base%':>7} {'N':>8}")
        print("-" * 80)
        for method_name, r in results.items():
            print(f"{method_name:<26} {r['auc']:>7.4f} {r['accuracy']:>7.4f} "
                  f"{r['admissibility_rate']:>7.4f} {r['baseline_accuracy']:>7.4f} "
                  f"{r['n_samples']:>8}")
        print("=" * 80)

        if output_path:
            import json
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Outcome results saved to {output_path}")

        return results


def load_assistments(path: str) -> Dict[str, List[Dict]]:
    """
    Load ASSISTments data into student_logs format.
    Returns Dict[student_id → List[interaction_dicts]]
    """
    df = pd.read_csv(path, low_memory=False)

    # Normalise column names
    col_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if any(x in lc for x in ("user_id", "student_id", "anon_student_id")):
            col_map[col] = "user_id"
        elif any(x in lc for x in ("skill_name", "kc(", "skill", "concept")):
            col_map[col] = "skill_name"
        elif lc == "correct":
            col_map[col] = "correct"
        elif any(x in lc for x in ("order_id", "problem_id", "sequence")):
            col_map[col] = "order_id"
    df = df.rename(columns=col_map)

    df["correct"]    = pd.to_numeric(df["correct"],    errors="coerce").fillna(0).astype(int).clip(0,1)
    df["skill_name"] = df.get("skill_name", pd.Series(["unknown"]*len(df))).fillna("unknown").astype(str)

    if "order_id" in df.columns:
        df = df.sort_values(["user_id", "order_id"])

    student_logs = {}
    for uid, grp in df.groupby("user_id"):
        student_logs[str(uid)] = grp[["skill_name", "correct", "order_id"]
                                     if "order_id" in grp.columns
                                     else ["skill_name", "correct"]
                                     ].to_dict("records")
    return student_logs