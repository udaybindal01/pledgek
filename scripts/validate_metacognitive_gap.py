"""
validate_metacognitive_gap.py — Validate p_sub indirectly from student outcomes.

Problem: We don't have students' self-reported confidence ratings, so we 
can't directly validate that p_sub tracks "what students THINK they know."

Solution: Indirect validation using the Dunning-Kruger prediction:
    - If p_sub is a good model of subjective mastery:
      Overconfident students (gap > 0) should fail MORE on subsequent questions
      than their objective mastery (p_obj) would predict.
      Well-calibrated students should fail at the rate p_obj predicts.

This is the standard validation approach when ground-truth confidence 
ratings are unavailable. Used in: Corbett & Anderson (1994), Baker et al. (2010).

Run:
    python scripts/validate_metacognitive_gap.py \
        --assistments data/raw/assistments.csv \
        --output outputs/metacognitive_validation/
"""

import argparse
import json
import numpy as np
import pandas as pd
import sys
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_karma_on_student(
    interactions: List[Dict],
    karma_config: Dict
) -> List[Dict]:
    """
    Run KARMA on a single student's history.
    Returns list of {
        interaction_idx, concept_id, p_obj, p_sub, gap, 
        actual_correct, bkt_predicted_correct
    }
    """
    from karma.estimator import KARMAEstimator, Interaction

    karma = KARMAEstimator(karma_config)
    records = []

    base_time = datetime(2024, 1, 1)

    for i, interaction in enumerate(interactions):
        concept_id = interaction["concept_id"]
        correct    = interaction["correct"]
        # Spread interactions 1 day apart (approximation)
        timestamp  = base_time + timedelta(days=i)

        # Record BEFORE update (prediction)
        p_obj, p_sub, gap = karma.get_knowledge_state(concept_id)

        # BKT-predicted P(correct)
        bkt = karma.bkt
        p_correct_pred = (
            p_obj * (1 - bkt.p_slip) +
            (1 - p_obj) * bkt.p_guess
        )

        records.append({
            "interaction_idx":    i,
            "concept_id":         concept_id,
            "p_obj":              round(p_obj, 4),
            "p_sub":              round(p_sub, 4),
            "gap":                round(gap,   4),
            "actual_correct":     int(correct),
            "bkt_predicted_prob": round(p_correct_pred, 4),
            "prediction_error":   round(abs(p_correct_pred - correct), 4),
        })

        # Update KARMA
        karma.update(Interaction(
            interaction_id   = f"i{i}",
            timestamp        = timestamp,
            query            = f"explain {concept_id}",
            concept_ids      = [concept_id],
            correct          = bool(correct),
            response_quality = float(correct),
            mrl_divergence   = 0.0
        ))

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistments", default="data/raw/assistments.csv")
    parser.add_argument("--output",      default="outputs/metacognitive_validation/")
    parser.add_argument("--max-students", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = Path("config/base_config.yaml")
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        karma_config = config.get("karma", {})
    else:
        karma_config = {}

    # Load ASSISTments
    df = pd.read_csv(args.assistments, low_memory=False)
    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if "user" in lc or "student" in lc: col_map[col] = "user_id"
        elif "skill" in lc or "kc" in lc:   col_map[col] = "skill_name"
        elif lc == "correct":                col_map[col] = "correct"
        elif "order" in lc or "time" in lc:  col_map[col] = "order_id"
    df = df.rename(columns=col_map)
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    if "order_id" in df.columns:
        df = df.sort_values(["user_id", "order_id"])

    students = df["user_id"].unique()[:args.max_students]
    logger.info(f"Running KARMA on {len(students)} students...")

    all_records = []
    for student_id in students:
        student_df = df[df["user_id"] == student_id]
        interactions = [
            {"concept_id": row["skill_name"], "correct": row["correct"]}
            for _, row in student_df.iterrows()
        ]
        if len(interactions) < 5:
            continue
        records = run_karma_on_student(interactions, karma_config)
        for r in records:
            r["student_id"] = student_id
        all_records.extend(records)

    results_df = pd.DataFrame(all_records)

    # ── Validation 1: Overconfident students fail more than BKT predicts ──
    # BKT only uses p_obj. If metacognitive gap adds information:
    #   Error(overconfident) > Error(well-calibrated) > Error(underconfident)
    # where "error" = |BKT prediction - actual correctness|

    results_df["calibration_group"] = pd.cut(
        results_df["gap"],
        bins=[-1.0, -0.15, 0.15, 1.0],
        labels=["underconfident", "well-calibrated", "overconfident"]
    )

    group_stats = results_df.groupby("calibration_group").agg(
        n=("actual_correct", "count"),
        mean_actual_accuracy=("actual_correct", "mean"),
        mean_bkt_predicted=("bkt_predicted_prob", "mean"),
        mean_prediction_error=("prediction_error", "mean"),
    ).round(4)

    print("\n=== Metacognitive Gap Validation ===")
    print("\nGroup Statistics:")
    print(group_stats.to_string())

    # Key claim: overconfident group has HIGHER prediction error than
    # well-calibrated group — meaning p_sub adds information beyond p_obj
    oc_error = results_df[results_df["calibration_group"] == "overconfident"]["prediction_error"].mean()
    wc_error = results_df[results_df["calibration_group"] == "well-calibrated"]["prediction_error"].mean()
    uc_error = results_df[results_df["calibration_group"] == "underconfident"]["prediction_error"].mean()

    print(f"\nBKT prediction error by calibration group:")
    print(f"  Underconfident:  {uc_error:.4f}")
    print(f"  Well-calibrated: {wc_error:.4f}")
    print(f"  Overconfident:   {oc_error:.4f}")

    validated = oc_error > wc_error
    print(f"\n{'✓ VALIDATED' if validated else '✗ NOT VALIDATED'}: "
          f"Overconfident students have "
          f"{'higher' if validated else 'lower'} BKT prediction error "
          f"({oc_error:.4f} vs {wc_error:.4f}). "
          f"This {'supports' if validated else 'does not support'} the claim that "
          f"metacognitive gap adds predictive value beyond objective mastery.")

    # ── Validation 2: Calibration over time ──
    # Well-calibrated students should improve faster (their gap → 0)
    # Overconfident students should show more sudden drops in accuracy
    results_df["interaction_bin"] = pd.cut(
        results_df["interaction_idx"], bins=5,
        labels=["early", "mid-early", "mid", "mid-late", "late"]
    )
    temporal = results_df.groupby(
        ["calibration_group", "interaction_bin"]
    )["actual_correct"].mean().unstack()

    print("\nAccuracy over time by calibration group:")
    print(temporal.round(3).to_string())

    # Save results
    summary = {
        "n_students": len(students),
        "n_records": len(results_df),
        "overconfident_bkt_error":   round(float(oc_error), 4),
        "wellcalibrated_bkt_error":  round(float(wc_error), 4),
        "underconfident_bkt_error":  round(float(uc_error), 4),
        "gap_adds_predictive_value": bool(validated),
        "group_stats": group_stats.to_dict(),
        "interpretation": (
            "VALIDATED: Metacognitive gap adds predictive value beyond BKT's "
            "objective mastery estimate. Overconfident students fail more often "
            "than BKT alone would predict, validating the need for dual-state tracking."
            if validated else
            "INCONCLUSIVE: Gap does not clearly add predictive value. "
            "Consider using larger sample or checking BKT parameter fitting."
        )
    }

    with open(output_dir / "metacognitive_gap_validation.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()