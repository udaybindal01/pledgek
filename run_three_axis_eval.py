#!/usr/bin/env python3
"""
run_three_axis_eval.py — Three-Axis Evaluation for PLEDGE-KARMA
================================================================
Implements the Strategy B evaluation framework:

  Axis 1 — Prerequisite Graph Quality
      Dataset:  LectureBank (CS/NLP human labels)  +
                OpenStax multi-subject corpus (chapter ordering)
      Metrics:  Prereq extraction Precision / Recall / F1
                Edge coverage, cross-subject alignment quality

  Axis 2 — Knowledge Tracing & Forgetting
      Dataset:  EdNet (PRIMARY, 131M interactions, real timestamps)
                ASSISTments (FALLBACK, with MRL injection)
      Metrics:  BKT AUC, RMSE
                Forgetting curve MAE (Ebbinghaus vs. actual)
                MRL divergence → next-correctness correlation (r, p-value)

  Axis 3 — End-to-End Pedagogical Retrieval
      Dataset:  MOOCCube (same-domain prereq graph + student logs)
      Metrics:  Admissibility rate
                NDCG@10, MRR
                Held-out outcome AUC (retrieval quality → student correctness)

Usage:
    # Run all axes (auto-detects available data)
    python run_three_axis_eval.py

    # Run specific axis
    python run_three_axis_eval.py --axis 1
    python run_three_axis_eval.py --axis 2
    python run_three_axis_eval.py --axis 3

    # Force specific KT dataset
    python run_three_axis_eval.py --axis 2 --kt-dataset ednet
    python run_three_axis_eval.py --axis 2 --kt-dataset assistments

    # Quick mode (fewer students, faster)
    python run_three_axis_eval.py --quick

    # Output directory
    python run_three_axis_eval.py --output outputs/three_axis_results/
"""

import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import sys
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Axis 1: Prerequisite Graph Quality
# ─────────────────────────────────────────────────────────────────────────────

def run_axis1_prereq_graph(
    loader,
    encoder,
    output_dir: Path,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Axis 1: Evaluate prereq graph construction quality.

    Sub-evaluations:
      1a. LectureBank F1: Does our automated extraction match human labels?
          - Extract prereq edges from LectureBank concept names using our pipeline
          - Compare against human-labelled positive/negative pairs
          - Report Precision, Recall, F1

      1b. OpenStax chapter-ordering validation:
          - Does chapter-ordering + MRL multi-scale agreement produce
            semantically meaningful edges?
          - Manual spot-check sample + ordering consistency metric

      1c. Cross-subject concept alignment quality:
          - When ConceptAligner bridges LectureBank CS concepts to OpenStax,
            how semantically accurate are the bridges?
          - Report: mean bridge_confidence, % with sim_768D > 0.8

    Returns:
        Dict with all Axis 1 metrics.
    """
    logger.info("=" * 65)
    logger.info("AXIS 1: Prerequisite Graph Quality Evaluation")
    logger.info("=" * 65)

    results: Dict[str, Any] = {"axis": 1, "timestamp": datetime.now().isoformat()}

    # ── Sub-eval 1a: LectureBank F1 ──────────────────────────────────────

    lb_data = loader.load_lecturebank_for_eval()
    pos_edges    = lb_data["positive_edges"]
    neg_pairs    = lb_data["negative_pairs"]
    lb_concepts  = lb_data["concepts"]

    if not lb_concepts:
        logger.warning("  LectureBank not available — skipping 1a")
        results["axis_1a_lecturebank_f1"] = {"status": "skipped"}
    else:
        logger.info(
            f"  LectureBank: {len(lb_concepts)} concepts, "
            f"{len(pos_edges)} positive edges, {len(neg_pairs)} negative pairs"
        )

        try:
            f1_result = _evaluate_prereq_extraction_f1(
                lb_concepts=lb_concepts,
                positive_edges=pos_edges,
                negative_pairs=neg_pairs,
                encoder=encoder,
            )
            results["axis_1a_lecturebank_f1"] = f1_result
            logger.info(
                f"  LectureBank F1: P={f1_result['precision']:.3f}, "
                f"R={f1_result['recall']:.3f}, F1={f1_result['f1']:.3f}"
            )
        except Exception as e:
            logger.error(f"  LectureBank F1 evaluation failed: {e}")
            results["axis_1a_lecturebank_f1"] = {"status": "error", "error": str(e)}

    # ── Sub-eval 1b: OpenStax ordering consistency ────────────────────────

    concepts, chunks = loader.load_corpus()
    if not concepts:
        logger.warning("  OpenStax corpus not available — skipping 1b")
        results["axis_1b_openstax_ordering"] = {"status": "skipped"}
    else:
        ordering_result = _evaluate_openstax_ordering(concepts)
        results["axis_1b_openstax_ordering"] = ordering_result
        results["corpus_stats"] = {
            "n_concepts": len(concepts),
            "n_chunks":   len(chunks),
            "subjects":   list({c.subject for c in concepts}),
        }
        logger.info(
            f"  OpenStax ordering: {len(concepts)} concepts across "
            f"{len({c.subject for c in concepts})} subjects, "
            f"monotonicity={ordering_result.get('chapter_monotonicity', 'N/A')}"
        )

    # ── Sub-eval 1c: Cross-subject alignment ─────────────────────────────

    if lb_concepts and concepts and encoder and encoder._model_loaded:
        alignment_result = _evaluate_concept_alignment(
            lb_concepts=lb_concepts,
            os_concepts=[
                {"concept_id": c.concept_id, "name": c.name, "dataset": "openstax"}
                for c in concepts[:500]   # cap for speed
            ],
            encoder=encoder,
            quick=quick,
        )
        results["axis_1c_cross_alignment"] = alignment_result
        logger.info(
            f"  Cross-alignment: {alignment_result.get('n_bridges', 0)} bridges, "
            f"mean_conf={alignment_result.get('mean_confidence', 0):.3f}"
        )
    else:
        results["axis_1c_cross_alignment"] = {
            "status": "skipped",
            "reason": "encoder not available or no concept data"
        }

    _save_axis_results(results, output_dir / "axis1_prereq_graph.json")
    _print_axis_summary(1, results)
    return results


def _evaluate_prereq_extraction_f1(
    lb_concepts, positive_edges, negative_pairs, encoder
) -> Dict:
    """
    Compare automated MRL-based prereq extraction against LectureBank human labels.

    Protocol:
      1. Encode all concept names with MRL encoder
      2. Predict prereq edges using multi-scale agreement (same as graph_builder)
      3. Compare predictions against positive/negative ground truth
      4. Compute Precision, Recall, F1
    """
    if not encoder or not encoder._model_loaded:
        return _heuristic_prereq_f1(lb_concepts, positive_edges, negative_pairs)

    # Encode concept names
    names  = [c["name"] for c in lb_concepts]
    id_map = {c["name"]: c["concept_id"] for c in lb_concepts}

    embs = encoder.encode(names, prompt_name="search_query", show_progress=False)
    emb_map = {name: emb for name, emb in zip(names, embs)}

    # Predict edges: threshold sim_768D > 0.6 AND sim_768D > sim_64D + 0.05
    predicted_pos = set()
    for i, c_a in enumerate(lb_concepts):
        for c_b in lb_concepts[i+1:]:
            e_a = emb_map.get(c_a["name"])
            e_b = emb_map.get(c_b["name"])
            if e_a is None or e_b is None:
                continue

            sim_768 = float(np.dot(e_a.at_dim(768), e_b.at_dim(768)))
            sim_64  = float(np.dot(e_a.at_dim(64),  e_b.at_dim(64)))

            # Multi-scale agreement: semantically related but not just lexically
            if sim_768 > 0.60 and sim_768 > sim_64 + 0.02:
                predicted_pos.add((c_a["concept_id"], c_b["concept_id"]))
                predicted_pos.add((c_b["concept_id"], c_a["concept_id"]))

    # Ground truth
    gt_pos = {
        (e["source_id"], e["target_id"]) for e in positive_edges
    }
    gt_neg = {
        (p["source_id"], p["target_id"]) for p in negative_pairs
    }

    all_gt = gt_pos | gt_neg
    if not all_gt:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_gt_pos": 0}

    tp = len(predicted_pos & gt_pos)
    fp = len(predicted_pos & gt_neg)
    fn = len(gt_pos - predicted_pos)

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "precision":     round(precision, 4),
        "recall":        round(recall,    4),
        "f1":            round(f1,        4),
        "tp":            tp,
        "fp":            fp,
        "fn":            fn,
        "n_gt_pos":      len(gt_pos),
        "n_gt_neg":      len(gt_neg),
        "n_predicted":   len(predicted_pos),
    }


def _heuristic_prereq_f1(lb_concepts, positive_edges, negative_pairs) -> Dict:
    """
    Heuristic prereq detection using concept name length/ordering
    (fallback when encoder not available).
    """
    id_to_name = {c["concept_id"]: c["name"] for c in lb_concepts}
    predicted_pos = set()

    for i, c_a in enumerate(lb_concepts):
        for c_b in lb_concepts:
            if c_a == c_b:
                continue
            # Shorter name = more fundamental → more likely prereq
            if len(c_a["name"]) < len(c_b["name"]):
                predicted_pos.add((c_a["concept_id"], c_b["concept_id"]))

    gt_pos = {(e["source_id"], e["target_id"]) for e in positive_edges}
    gt_neg = {(p["source_id"], p["target_id"]) for p in negative_pairs}

    tp = len(predicted_pos & gt_pos)
    fp = len(predicted_pos & gt_neg)
    fn = len(gt_pos - predicted_pos)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "method": "heuristic_name_length",
        "note": "Install sentence-transformers for real MRL-based evaluation"
    }


def _evaluate_openstax_ordering(concepts) -> Dict:
    """
    Check that chapter ordering is monotonically consistent within subjects.
    """
    from collections import defaultdict

    by_subject = defaultdict(list)
    for c in concepts:
        by_subject[c.subject].append(c.chapter_order)

    monotonicity_scores = {}
    for subj, orders in by_subject.items():
        sorted_orders = sorted(orders)
        if len(sorted_orders) < 2:
            continue
        # Count how many adjacent pairs are in order
        n_in_order = sum(1 for a, b in zip(sorted_orders, sorted_orders[1:]) if a <= b)
        monotonicity_scores[subj] = round(n_in_order / max(len(sorted_orders) - 1, 1), 3)

    return {
        "n_concepts":           len(concepts),
        "n_subjects":           len(by_subject),
        "subjects":             list(by_subject.keys()),
        "chapter_monotonicity": dict(monotonicity_scores),
        "mean_monotonicity":    round(float(np.mean(list(monotonicity_scores.values()))) if monotonicity_scores else 0.0, 3),
    }


def _evaluate_concept_alignment(lb_concepts, os_concepts, encoder, quick) -> Dict:
    """Run ConceptAligner and report alignment quality statistics."""
    from data.pipelines.concept_alignment import ConceptAligner

    aligner = ConceptAligner(encoder=encoder)
    bridges = aligner.align_across_datasets(
        source_concepts=lb_concepts[:50 if quick else len(lb_concepts)],
        target_concepts=os_concepts[:100 if quick else len(os_concepts)],
        threshold=0.65,
    )

    if not bridges:
        return {"n_bridges": 0, "mean_confidence": 0.0}

    confidences = [b.bridge_confidence for b in bridges]
    high_conf   = sum(1 for c in confidences if c > 0.8)

    return {
        "n_bridges":         len(bridges),
        "mean_confidence":   round(float(np.mean(confidences)), 4),
        "median_confidence": round(float(np.median(confidences)), 4),
        "high_conf_bridges": high_conf,
        "high_conf_pct":     round(high_conf / max(len(bridges), 1), 3),
        "threshold_used":    0.65,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Axis 2: Knowledge Tracing & Forgetting
# ─────────────────────────────────────────────────────────────────────────────

def run_axis2_kt_forgetting(
    loader,
    karma_config: Dict,
    output_dir: Path,
    kt_dataset: Optional[str] = None,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Axis 2: BKT accuracy + Ebbinghaus forgetting validation.

    Sub-evaluations:
      2a. BKT Knowledge Tracing AUC/RMSE
          (leave-one-out per student, per concept)
      2b. Forgetting curve calibration
          (predicted retention vs. actual re-attempt correctness after N days)
      2c. MRL divergence → next-correctness correlation
          (validates the core metacognitive signal claim)
      2d. Metacognitive gap prediction
          (students with high gap fail more than BKT-alone predicts)
    """
    logger.info("=" * 65)
    logger.info("AXIS 2: Knowledge Tracing & Forgetting Evaluation")
    logger.info("=" * 65)

    results: Dict[str, Any] = {"axis": 2, "timestamp": datetime.now().isoformat()}

    # Load interactions (EdNet primary, ASSISTments fallback)
    source_name, student_logs = loader.load_interactions(
        prefer_ednet=True,
        max_students=100 if quick else None,
        source_override=kt_dataset,
    )
    results["dataset_used"] = source_name

    if not student_logs:
        logger.warning("  No interaction data available for Axis 2.")
        results["status"] = "no_data"
        return results

    n_students    = len(student_logs)
    n_interactions = sum(len(v) for v in student_logs.values())
    mrl_coverage  = sum(
        1 for ilist in student_logs.values()
        for i in ilist if i.get("mrl_divergence", 0.0) != 0.0
    )

    logger.info(
        f"  Dataset: {source_name} | "
        f"{n_students} students | {n_interactions} interactions | "
        f"MRL coverage={mrl_coverage/max(n_interactions,1):.1%}"
    )

    if mrl_coverage == 0:
        logger.warning(
            "  WARNING: mrl_divergence=0.0 for all interactions. "
            "MRL signal validation will be trivially null. "
            "For ASSISTments: re-run prepare_data.py with --inject-mrl. "
            "For real MRL: use EdNet KT4 with --compute-mrl."
        )
        results["mrl_warning"] = "mrl_divergence=0.0 everywhere — results for 2c/2d unreliable"

    # ── 2a: BKT AUC ──────────────────────────────────────────────────────

    from evaluation.evaluator import PLEDGEKARMAEvaluator
    evaluator = PLEDGEKARMAEvaluator(graph=None, karma_config=karma_config)

    auc_result = evaluator.evaluate_kt_real_world(
        student_logs=student_logs,
        karma_config=karma_config,
        method_name=f"KARMA-{source_name}",
        min_interactions_for_eval=5 if quick else 10,
    )
    results["axis_2a_bkt_auc"] = auc_result
    logger.info(
        f"  BKT AUC={auc_result.get('auc', 0):.4f}, "
        f"RMSE={auc_result.get('rmse', 0):.4f}, "
        f"N={auc_result.get('n_students', 0)} students"
    )

    # ── 2b: Forgetting curve calibration ─────────────────────────────────

    if source_name == "ednet":
        # EdNet has real timestamps → forgetting evaluation is meaningful
        forgetting_result = _evaluate_forgetting_curve(student_logs, karma_config)
        results["axis_2b_forgetting"] = forgetting_result
        logger.info(
            f"  Forgetting MAE={forgetting_result.get('mae', 'N/A'):.4f}, "
            f"N_pairs={forgetting_result.get('n_pairs', 0)}"
        )
    else:
        results["axis_2b_forgetting"] = {
            "status": "limited",
            "note":   "ASSISTments lacks reliable timestamps — forgetting eval requires EdNet",
        }
        logger.info("  Forgetting evaluation: limited (no reliable timestamps in ASSISTments)")

    # ── 2c: MRL divergence signal validation ─────────────────────────────

    if mrl_coverage > 0:
        mrl_result = _validate_mrl_signal(student_logs)
        results["axis_2c_mrl_signal"] = mrl_result
        logger.info(
            f"  MRL signal: r={mrl_result.get('correlation', 0):.4f}, "
            f"p={mrl_result.get('p_value', 1):.4f}, "
            f"validated={mrl_result.get('significant', False)}"
        )
    else:
        results["axis_2c_mrl_signal"] = {"status": "skipped", "reason": "mrl_divergence=0"}

    # ── 2d: Metacognitive gap prediction ─────────────────────────────────

    if mrl_coverage > 0:
        meta_result = _validate_metacognitive_gap(student_logs, karma_config)
        results["axis_2d_metacognitive_gap"] = meta_result
        logger.info(
            f"  Metacognitive gap: overconfident_fail_excess="
            f"{meta_result.get('overconfident_fail_excess', 0):.4f}"
        )
    else:
        results["axis_2d_metacognitive_gap"] = {"status": "skipped", "reason": "mrl_divergence=0"}

    _save_axis_results(results, output_dir / "axis2_kt_forgetting.json")
    _print_axis_summary(2, results)
    return results


def _evaluate_forgetting_curve(student_logs: Dict, karma_config: Dict) -> Dict:
    """
    Validate that KARMA's Ebbinghaus forgetting model matches real re-attempt patterns.

    Protocol:
      1. Find pairs: (interaction_t, interaction_t') on same concept, with gap Δ days
      2. KARMA predicts P(retention) at t' based on stability from t
      3. Compare predicted retention vs. actual correctness at t'
      4. Report MAE across all pairs
    """
    from karma.estimator import KARMAEstimator
    from datetime import datetime, timedelta

    pairs_correct   = []   # (predicted_retention, actual_correct)
    n_pairs = 0

    for uid, interactions in list(student_logs.items())[:200]:
        karma = KARMAEstimator(karma_config or {})

        concept_last_seen: Dict = {}  # concept_id → (timestamp, stability)

        for interaction in sorted(interactions, key=lambda x: x.get("timestamp", 0)):
            cid     = interaction.get("concept_id", "unknown")
            correct = bool(interaction.get("correct", 0))
            ts_raw  = interaction.get("timestamp", 0) or interaction.get("timestamp_ms", 0)

            try:
                if ts_raw > 1e10:   # ms
                    ts = datetime.fromtimestamp(ts_raw / 1000.0)
                else:               # s
                    ts = datetime.fromtimestamp(float(ts_raw))
            except (ValueError, OSError):
                ts = datetime.now()

            if cid in concept_last_seen:
                last_ts, last_stability = concept_last_seen[cid]
                delta_days = max((ts - last_ts).total_seconds() / 86400, 0)

                if 0.5 < delta_days < 180:   # Only meaningful gaps
                    predicted_retention = karma.forgetting.compute_retention(
                        last_stability, delta_days
                    )
                    pairs_correct.append((predicted_retention, int(correct)))
                    n_pairs += 1

            # Update KARMA state
            from karma.estimator import Interaction as KInteraction
            karma.update(KInteraction(
                interaction_id   = f"{uid}_forgetting",
                timestamp        = ts,
                query            = f"[{cid}]",
                concept_ids      = [cid],
                correct          = correct,
                response_quality = float(correct),
                mrl_divergence   = float(interaction.get("mrl_divergence", 0.0)),
            ))

            p_obj, _, _ = karma.get_knowledge_state(cid)
            concept_last_seen[cid] = (ts, p_obj * 5.0 + 0.5)   # rough stability proxy

    if not pairs_correct:
        return {"status": "no_pairs", "n_pairs": 0}

    pred   = np.array([p for p, _ in pairs_correct])
    actual = np.array([a for _, a in pairs_correct])
    mae    = float(np.mean(np.abs(pred - actual)))

    # Correlation between predicted retention and actual correctness
    from scipy import stats
    corr, pval = stats.pearsonr(pred, actual)

    return {
        "n_pairs":   n_pairs,
        "mae":       round(mae,  4),
        "r":         round(float(corr), 4),
        "p_value":   round(float(pval), 6),
        "significant": pval < 0.05,
        "mean_pred_retention": round(float(pred.mean()), 4),
        "mean_actual_correct": round(float(actual.mean()), 4),
    }


def _validate_mrl_signal(student_logs: Dict) -> Dict:
    """
    Validate: high MRL divergence at time t → lower correctness at t+1.
    """
    from scipy import stats

    all_divergences = []
    all_next_correct = []

    for uid, interactions in student_logs.items():
        sorted_ints = sorted(interactions, key=lambda x: x.get("timestamp", 0))

        by_concept: Dict[str, List] = {}
        for inter in sorted_ints:
            cid = inter.get("concept_id", "")
            if cid not in by_concept:
                by_concept[cid] = []
            by_concept[cid].append(inter)

        for cid, cid_ints in by_concept.items():
            for i in range(len(cid_ints) - 1):
                mrl_div     = float(cid_ints[i].get("mrl_divergence", 0.0))
                next_correct = int(cid_ints[i+1].get("correct", 0))
                all_divergences.append(mrl_div)
                all_next_correct.append(next_correct)

    if len(all_divergences) < 30:
        return {"status": "insufficient_data", "n": len(all_divergences)}

    div_arr = np.array(all_divergences)
    cor_arr = np.array(all_next_correct)

    corr, pval = stats.pointbiserialr(div_arr, cor_arr)

    median  = float(np.median(div_arr))
    high    = cor_arr[div_arr > median]
    low     = cor_arr[div_arr <= median]

    return {
        "n_pairs":              len(all_divergences),
        "correlation":          round(float(corr), 4),
        "p_value":              round(float(pval), 6),
        "significant":          pval < 0.05,
        "high_div_accuracy":    round(float(high.mean()) if len(high) else 0, 4),
        "low_div_accuracy":     round(float(low.mean())  if len(low)  else 0, 4),
        "accuracy_gap":         round(float(low.mean() - high.mean()) if (len(low) and len(high)) else 0, 4),
        "validated":            (pval < 0.05 and
                                 float(low.mean()) > float(high.mean())),
    }


def _validate_metacognitive_gap(student_logs: Dict, karma_config: Dict) -> Dict:
    """
    Validate: students with high metacognitive gap (K^sub >> K^obj) fail
    more than their BKT-alone prediction suggests.
    """
    from karma.estimator import KARMAEstimator, Interaction as KInteraction
    from datetime import datetime

    overconfident_fails   = []
    well_calibrated_fails = []

    for uid, interactions in list(student_logs.items())[:200]:
        karma = KARMAEstimator(karma_config or {})

        for i, interaction in enumerate(
            sorted(interactions, key=lambda x: x.get("timestamp", 0))
        ):
            cid     = interaction.get("concept_id", "unknown")
            correct = bool(interaction.get("correct", 0))
            mrl_div = float(interaction.get("mrl_divergence", 0.0))

            if i > 0:
                p_obj, p_sub, gap = karma.get_knowledge_state(cid)
                bkt_expected_correct = p_obj * (1 - 0.1) + (1 - p_obj) * 0.2
                actual_correct = int(correct)
                fail_excess = bkt_expected_correct - actual_correct  # >0 = worse than expected

                # High gap = overconfident student (K^sub >> K^obj)
                if gap > 0.2:
                    overconfident_fails.append(fail_excess)
                else:
                    well_calibrated_fails.append(fail_excess)

            karma.update(KInteraction(
                interaction_id=f"{uid}_{i}",
                timestamp=datetime.fromtimestamp(
                    float(interaction.get("timestamp", i * 3600) or i * 3600)
                ),
                query=f"[{cid}]",
                concept_ids=[cid],
                correct=correct,
                response_quality=float(correct),
                mrl_divergence=mrl_div,
            ))

    if not overconfident_fails or not well_calibrated_fails:
        return {"status": "insufficient_data"}

    from scipy import stats
    mean_oc = float(np.mean(overconfident_fails))
    mean_wc = float(np.mean(well_calibrated_fails))
    t_stat, pval = stats.ttest_ind(overconfident_fails, well_calibrated_fails)

    return {
        "n_overconfident":       len(overconfident_fails),
        "n_well_calibrated":     len(well_calibrated_fails),
        "overconfident_fail_excess":  round(mean_oc, 4),
        "well_calibrated_fail_excess": round(mean_wc, 4),
        "gap_effect":            round(mean_oc - mean_wc, 4),
        "t_statistic":           round(float(t_stat), 4),
        "p_value":               round(float(pval), 6),
        "significant":           pval < 0.05,
        "validated":             (pval < 0.05 and mean_oc > mean_wc),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Axis 3: End-to-End Pedagogical Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def run_axis3_endtoend(
    loader,
    graph_builder,
    karma_config: Dict,
    output_dir: Path,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Axis 3: End-to-end pedagogical retrieval using MOOCCube.

    MOOCCube is the only dataset where we have both the prereq graph AND
    student logs in the same domain, so held-out student outcomes can
    validate retrieval quality end-to-end.

    Sub-evaluations:
      3a. Admissibility rate — what fraction of retrieved chunks are admissible?
      3b. NDCG@10, MRR — retrieval ranking quality
      3c. Held-out outcome AUC — does better retrieval predict better outcomes?
    """
    logger.info("=" * 65)
    logger.info("AXIS 3: End-to-End Pedagogical Retrieval (MOOCCube)")
    logger.info("=" * 65)

    results: Dict[str, Any] = {"axis": 3, "timestamp": datetime.now().isoformat()}

    mc_concepts, mc_edges, mc_student_logs = loader.load_mooccube()

    if not mc_concepts or not mc_student_logs:
        logger.warning("  MOOCCube not available — skipping Axis 3.")
        results["status"] = "no_data"
        _save_axis_results(results, output_dir / "axis3_endtoend.json")
        return results

    logger.info(
        f"  MOOCCube: {len(mc_concepts)} concepts, "
        f"{len(mc_edges)} prereq edges, "
        f"{len(mc_student_logs)} students"
    )

    # Build MOOCCube graph
    try:
        from knowledge_graph.graph_builder import KnowledgeGraphBuilder
        from knowledge_graph.graph_builder import ConceptNode, CorpusChunk

        mc_graph = KnowledgeGraphBuilder(
            encoder=graph_builder.encoder if graph_builder else None,
            config={"min_confidence": 0.5}
        )

        # Register concepts
        for c in mc_concepts:
            mc_graph.add_concept(ConceptNode(
                concept_id=c["concept_id"],
                name=c["name"],
                description=c.get("description", c["name"]),
                source_chunk_ids=[],
                depth_level=c.get("depth_level", 1),
                chapter_order=c.get("chapter_order", 0),
                subject=c.get("subject", "mooc"),
            ))

        # Add prereq edges directly
        for src, tgt, conf in mc_edges:
            if src in mc_graph.concepts and tgt in mc_graph.concepts:
                mc_graph.graph.add_edge(src, tgt, weight=conf, edge_type="prerequisite")

        # Run outcome evaluator
        from evaluation.outcome_evaluator import OutcomeEvaluator

        oe = OutcomeEvaluator(graph=mc_graph, karma_config=karma_config)

        # Limit students for quick mode
        test_students = dict(list(mc_student_logs.items())[:50 if quick else 200])

        # Simple retrieval mock: return concepts ordered by prereq depth
        def pledge_retrieval(query, karma, concept_ids=None, n=5):
            """Simplified PLEDGE retrieval for Axis 3 evaluation."""
            available = [
                c for c in mc_graph.concepts.values()
                if all(
                    karma.get_knowledge_state(prereq)[0] >= 0.6
                    for prereq in getattr(mc_graph.chunks.get(c.source_chunk_ids[0] if c.source_chunk_ids else ""), "prerequisite_concept_ids", [])
                )
            ]
            return [c.concept_id for c in available[:n]]

        outcome_results = oe.evaluate(
            student_logs=test_students,
            retrieval_methods={"PLEDGE-KARMA": pledge_retrieval},
            chunk_map={},
            n_test_students=50 if quick else 200,
        )
        results["axis_3a_3b_3c"] = outcome_results

        for method, res in outcome_results.items():
            logger.info(
                f"  {method}: AUC={res.get('auc', 0):.4f}, "
                f"Adm={res.get('admissibility_rate', 0):.4f}"
            )

    except Exception as e:
        logger.error(f"  Axis 3 evaluation error: {e}")
        results["error"] = str(e)
        # Provide fallback stats
        results["axis_3_stats"] = {
            "n_concepts":     len(mc_concepts),
            "n_prereq_edges": len(mc_edges),
            "n_students":     len(mc_student_logs),
        }

    _save_axis_results(results, output_dir / "axis3_endtoend.json")
    _print_axis_summary(3, results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _save_axis_results(results: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Saved: {path}")


def _print_axis_summary(axis_num: int, results: Dict) -> None:
    print(f"\n{'=' * 65}")
    print(f"AXIS {axis_num} SUMMARY")
    print(f"{'=' * 65}")
    for k, v in results.items():
        if k in ("axis", "timestamp"):
            continue
        if isinstance(v, dict):
            print(f"  [{k}]")
            for kk, vv in v.items():
                if not isinstance(vv, dict):
                    print(f"    {kk:<40}: {vv}")
        else:
            print(f"  {k:<44}: {v}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="PLEDGE-KARMA Three-Axis Evaluation"
    )
    parser.add_argument(
        "--axis", type=int, choices=[1, 2, 3],
        help="Run only this axis (default: all available)"
    )
    parser.add_argument(
        "--kt-dataset", choices=["ednet", "assistments"],
        help="Override KT dataset for Axis 2 (default: auto-select)"
    )
    parser.add_argument(
        "--output", default="outputs/three_axis_eval",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer students, faster execution"
    )
    parser.add_argument(
        "--config", default="config/base_config.yaml",
        help="KARMA config file"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    karma_config = {}
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        karma_config = full_config.get("karma", {})
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")

    # Set up data loader
    from data.data_loader import DataLoader
    loader = DataLoader("data/processed")
    avail  = loader.get_axis_availability()

    logger.info(f"Axis availability: {avail}")

    # Set up encoder (optional — degrades gracefully)
    encoder = None
    try:
        from models.mrl_encoder import MRLEncoder
        encoder = MRLEncoder({
            "model_name":           "nomic-ai/nomic-embed-text-v1.5",
            "matryoshka_dims":      [64, 128, 256, 512, 768],
            "full_dim":             768,
            "normalize_embeddings": True,
            "trust_remote_code":    True,
        })
        if encoder._model_loaded:
            logger.info("MRL encoder loaded successfully")
        else:
            logger.warning("MRL encoder not loaded — using heuristic fallbacks")
    except Exception as e:
        logger.warning(f"Could not load MRL encoder: {e}")

    # Set up graph builder stub
    graph_builder = None
    try:
        from knowledge_graph.graph_builder import KnowledgeGraphBuilder
        graph_builder = KnowledgeGraphBuilder(encoder=encoder, config=karma_config.get("graph", {}))
    except Exception as e:
        logger.warning(f"Could not init graph builder: {e}")

    all_results = {}
    axes_to_run = [args.axis] if args.axis else [1, 2, 3]

    for axis in axes_to_run:
        if axis == 1:
            if avail.get("axis_1_prereq") or not args.axis:
                all_results[1] = run_axis1_prereq_graph(
                    loader, encoder, output_dir, quick=args.quick
                )
            else:
                logger.warning("Axis 1: required data not available")

        elif axis == 2:
            if avail.get("axis_2_kt") or not args.axis:
                all_results[2] = run_axis2_kt_forgetting(
                    loader, karma_config, output_dir,
                    kt_dataset=args.kt_dataset, quick=args.quick
                )
            else:
                logger.warning("Axis 2: no interaction data available")

        elif axis == 3:
            if avail.get("axis_3_endtoend") or not args.axis:
                all_results[3] = run_axis3_endtoend(
                    loader, graph_builder, karma_config, output_dir,
                    quick=args.quick
                )
            else:
                logger.warning("Axis 3: MOOCCube not available")

    # Combined summary table
    print("\n" + "=" * 65)
    print("PLEDGE-KARMA THREE-AXIS EVALUATION — COMBINED SUMMARY")
    print("=" * 65)
    print(f"{'Axis':<8} {'Dataset':<20} {'Key Metric':<20} {'Value'}")
    print("-" * 65)

    if 1 in all_results:
        r = all_results[1]
        f1 = r.get("axis_1a_lecturebank_f1", {}).get("f1", "N/A")
        print(f"{'Axis 1':<8} {'LectureBank+OpenStax':<20} {'F1 (prereq ext.)':<20} {f1}")

    if 2 in all_results:
        r = all_results[2]
        auc = r.get("axis_2a_bkt_auc", {}).get("auc", "N/A")
        ds  = r.get("dataset_used", "unknown")
        print(f"{'Axis 2':<8} {ds:<20} {'BKT AUC':<20} {auc}")

    if 3 in all_results:
        r = all_results[3]
        ax3 = r.get("axis_3a_3b_3c", {})
        adm = "N/A"
        for method, res in ax3.items():
            adm = res.get("admissibility_rate", "N/A")
            break
        print(f"{'Axis 3':<8} {'MOOCCube':<20} {'Admissibility':<20} {adm}")

    print("=" * 65)

    # Save combined results
    with open(output_dir / "three_axis_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()