#!/usr/bin/env python3
"""
PLEDGE-KARMA Data Preparation Script
=======================================
Downloads and processes all datasets needed for the research.

Three-Axis Evaluation Strategy (Strategy B + Strategy C):
  Axis 1 — Prereq Graph Quality:
      LectureBank (CS/NLP human labels)  +  OpenStax multi-subject corpus
  Axis 2 — KT / Forgetting:
      EdNet (preferred, 131M interactions, real timestamps + concept tags)
      ASSISTments (fallback, with MRL injection via ConceptAligner)
  Axis 3 — End-to-End Pedagogical Retrieval:
      MOOCCube (prereq graph + student logs in same domain)

Usage:
    # Quick mode: first 5 OpenStax chapters + 50K ASSISTments rows
    python data/prepare_data.py --mode quick

    # Full mode: all available data
    python data/prepare_data.py --mode full

    # Specific datasets
    python data/prepare_data.py --only openstax        # full multi-subject corpus
    python data/prepare_data.py --only lecturebank
    python data/prepare_data.py --only assistments     # with MRL injection
    python data/prepare_data.py --only ednet           # preferred KT dataset
    python data/prepare_data.py --only mooccube
    python data/prepare_data.py --only junyi

    # EdNet with simulated data (no download required)
    python data/prepare_data.py --only ednet --ednet-simulate
"""

import os
import sys
import json
import csv
import logging
import hashlib
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ═════════════════════════════════════════════════════════════════════════════
# 1. OpenStax  — Multi-subject corpus (Axis 1: prereq graph + retrieval target)
# ═════════════════════════════════════════════════════════════════════════════

def prepare_openstax(max_chapters: Optional[int] = None, multi_subject: bool = True):
    """
    Download and process OpenStax dataset from HuggingFace.

    In full mode, processes ALL subjects (physics, chemistry, calculus,
    biology, statistics) to give the prereq graph a multi-domain corpus.
    This is the Axis 1 corpus for prereq graph evaluation.

    In quick mode, processes physics only (1 book).
    """
    logger.info("=" * 60)
    logger.info("OPENSTAX: Downloading HuggingFaceTB/openstax_paragraphs...")
    if multi_subject and max_chapters is None:
        logger.info("  Mode: FULL multi-subject corpus (all subjects)")
    else:
        logger.info("  Mode: quick (physics only)")
    logger.info("=" * 60)

    from data.pipelines.hf_openstax_pipeline import HFOpenStaxPipeline

    pipeline = HFOpenStaxPipeline(output_dir=str(PROCESSED_DIR))

    try:
        if max_chapters is not None:
            # Quick mode: physics only, 1 book
            concepts, chunks = pipeline.process_dataset(
                max_books=1, book_name_filter="physics"
            )
            pipeline.save_processed("openstax_full")
        else:
            # Full mode: all subjects, no book filter
            concepts, chunks = pipeline.process_dataset(
                max_books=None, book_name_filter=""
            )
            pipeline.save_processed("openstax_full")

        logger.info(f"✓ OpenStax (HF): {len(concepts)} concepts, {len(chunks)} chunks")

        # Log subject distribution for paper reporting
        from collections import Counter
        subject_counts = Counter(c.subject for c in concepts)
        logger.info("  Subject distribution:")
        for subj, count in sorted(subject_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {subj:<20}: {count} concepts")

        return True

    except Exception as e:
        logger.error(f"  OpenStax download failed: {e}")
        logger.info("  Creating simulated multi-subject OpenStax corpus for testing...")
        _create_simulated_openstax_multisubject()
        return True


def _create_simulated_openstax_multisubject():
    """
    Create simulated multi-subject OpenStax corpus.
    Covers physics, calculus, chemistry, and CS to test cross-subject
    prereq alignment with LectureBank.
    """
    from experiments.run_experiment import build_mock_corpus

    out_dir = PROCESSED_DIR / "openstax_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-subject mocks and merge
    all_concepts = []
    all_chunks   = []

    subjects = [
        ("physics",   30, 100),
        ("calculus",  20,  80),
        ("chemistry", 20,  80),
        ("cs",        15,  60),   # Overlaps with LectureBank CS concepts
    ]

    for subject, n_concepts, n_chunks in subjects:
        concepts_obj, chunks_obj = build_mock_corpus(
            n_concepts=n_concepts, n_chunks=n_chunks
        )
        # Tag with subject
        for c in concepts_obj:
            c.subject = subject
            c.tags    = ["openstax", subject]
        for ch in chunks_obj:
            ch.subject = subject
            ch.source  = f"openstax_{subject}"

        all_concepts.extend(concepts_obj)
        all_chunks.extend(chunks_obj)

    concepts_list = []
    for c in all_concepts:
        concepts_list.append({
            "concept_id":      c.concept_id,
            "name":            c.name,
            "description":     c.description,
            "depth_level":     c.depth_level,
            "chapter_order":   c.chapter_order,
            "subject":         c.subject,
            "tags":            c.tags,
            "source_chunk_ids": c.source_chunk_ids,
        })

    chunks_list = []
    for ch in all_chunks:
        chunks_list.append({
            "chunk_id":                 ch.chunk_id,
            "text":                     ch.text,
            "concept_ids":              ch.concept_ids,
            "prerequisite_concept_ids": ch.prerequisite_concept_ids,
            "depth_level":              ch.depth_level,
            "chapter_order":            ch.chapter_order,
            "subject":                  ch.subject,
            "source":                   ch.source,
            "metadata":                 ch.metadata,
        })

    with open(out_dir / "concepts.json", "w") as f:
        json.dump(concepts_list, f, indent=2)
    with open(out_dir / "chunks.json", "w") as f:
        json.dump(chunks_list, f, indent=2)

    logger.info(
        f"  ✓ Simulated multi-subject OpenStax: "
        f"{len(concepts_list)} concepts, {len(chunks_list)} chunks "
        f"across {len(subjects)} subjects"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. LectureBank  — Prereq graph ground truth (Axis 1)
# ═════════════════════════════════════════════════════════════════════════════

LECTUREBANK_REPO = "https://github.com/Yale-LILY/LectureBank.git"


def prepare_lecturebank():
    """Clone LectureBank repo and extract prerequisite labels."""
    logger.info("=" * 60)
    logger.info("LECTUREBANK: Cloning prerequisite graph data (Axis 1)...")
    logger.info("=" * 60)

    lb_raw = RAW_DIR / "lecturebank"
    lb_out = PROCESSED_DIR / "lecturebank"
    lb_out.mkdir(parents=True, exist_ok=True)

    if not (lb_raw / ".git").exists():
        lb_raw.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", LECTUREBANK_REPO, str(lb_raw)],
                check=True, capture_output=True, text=True, timeout=120
            )
            logger.info("  Cloned LectureBank repository")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"  Git clone failed: {e}")
            logger.info("  Falling back to simulated LectureBank data")
            _create_simulated_lecturebank(lb_out)
            return True
    else:
        logger.info("  LectureBank already cloned")

    # Search for prerequisite annotation files
    prereq_files = list(lb_raw.rglob("*.txt"))
    concept_files = list(lb_raw.rglob("*concept*"))

    prereq_edges = []
    concepts_found = set()

    for pf in prereq_files:
        if "prereq" in pf.name.lower() or "prerequisite" in pf.name.lower():
            with open(pf, errors="ignore") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        concept_a, concept_b, label = parts[0], parts[1], parts[2]
                        concepts_found.update([concept_a, concept_b])
                        if label.strip() == "1":
                            cid_a = hashlib.md5(concept_a.encode()).hexdigest()[:12]
                            cid_b = hashlib.md5(concept_b.encode()).hexdigest()[:12]
                            prereq_edges.append({
                                "source_id":   f"lb_{cid_a}",
                                "target_id":   f"lb_{cid_b}",
                                "source_name": concept_a,
                                "target_name": concept_b,
                                "confidence":  0.9,
                                "source":      "lecturebank_human",
                            })

    if not prereq_edges:
        logger.warning("  No prereq edges found in LectureBank files.")
        logger.info("  Using simulated LectureBank data instead.")
        _create_simulated_lecturebank(lb_out)
        return True

    with open(lb_out / "prereq_edges.json", "w") as f:
        json.dump(prereq_edges, f, indent=2)

    # Save concept list for cross-dataset alignment
    concept_list = [
        {
            "concept_id": f"lb_{hashlib.md5(c.encode()).hexdigest()[:12]}",
            "name": c,
            "dataset": "lecturebank",
            "subject": "cs_nlp",
        }
        for c in sorted(concepts_found)
    ]
    with open(lb_out / "concepts.json", "w") as f:
        json.dump(concept_list, f, indent=2)

    logger.info(
        f"✓ LectureBank: {len(prereq_edges)} prereq edges, "
        f"{len(concepts_found)} concepts"
    )
    return True


def _create_simulated_lecturebank(lb_out: Path):
    """Simulated LectureBank with CS + NLP concepts (two chains)."""
    lb_out.mkdir(parents=True, exist_ok=True)

    concept_names = [
        "variables", "data types", "control flow", "functions", "recursion",
        "arrays", "linked lists", "trees", "graphs", "sorting",
        "dynamic programming", "gradient descent", "backpropagation",
        "neural networks", "word embeddings", "attention", "transformers",
        "language models",
    ]
    prereq_pairs = [
        ("variables", "data types"), ("data types", "control flow"),
        ("control flow", "functions"), ("functions", "recursion"),
        ("arrays", "sorting"), ("arrays", "linked lists"),
        ("linked lists", "trees"), ("trees", "graphs"),
        ("sorting", "dynamic programming"),
        ("gradient descent", "backpropagation"),
        ("backpropagation", "neural networks"),
        ("word embeddings", "attention"), ("attention", "transformers"),
        ("transformers", "language models"),
    ]
    neg_pairs = [
        ("recursion", "transformers"), ("sorting", "attention"),
        ("variables", "language models"), ("linked lists", "gradient descent"),
        ("arrays", "transformers"),
    ]

    def cid(name):
        return f"lb_{hashlib.md5(name.encode()).hexdigest()[:12]}"

    prereq_edges = [
        {
            "source_id":   cid(a), "target_id":   cid(b),
            "source_name": a,       "target_name": b,
            "confidence":  0.9,     "source":      "lb_simulated_human",
        }
        for a, b in prereq_pairs
    ]
    with open(lb_out / "prereq_edges.json", "w") as f:
        json.dump(prereq_edges, f, indent=2)

    concept_list = [
        {"concept_id": cid(n), "name": n, "dataset": "lecturebank", "subject": "cs_nlp"}
        for n in concept_names
    ]
    with open(lb_out / "concepts.json", "w") as f:
        json.dump(concept_list, f, indent=2)

    # Save negative pairs for evaluation (important for precision/recall)
    neg_pairs_list = [
        {"source_id": cid(a), "target_id": cid(b), "label": 0}
        for a, b in neg_pairs
    ]
    with open(lb_out / "negative_pairs.json", "w") as f:
        json.dump(neg_pairs_list, f, indent=2)

    logger.info(
        f"  ✓ Simulated LectureBank: {len(prereq_edges)} prereq edges, "
        f"{len(concept_list)} concepts"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3. ASSISTments  — KT fallback (Axis 2, with MRL injection)
# ═════════════════════════════════════════════════════════════════════════════

ASSISTMENTS_DIRECT_URL = (
    "https://raw.githubusercontent.com/hcnoh/knowledge-tracing-collection-tensorflow2.0/"
    "master/data/assistments09/data.csv"
)


def prepare_assistments(max_rows: Optional[int] = None, inject_mrl: bool = False):
    """
    Download and process ASSISTments dataset.

    ASSISTments is Axis 2 FALLBACK for KT/forgetting evaluation when EdNet
    is not available. Use EdNet when possible (real timestamps + MRL divergence).

    inject_mrl: If True, retroactively compute MRL divergence from skill name
                templates using the ConceptAligner (Strategy C). Requires
                sentence-transformers to be installed.
    """
    logger.info("=" * 60)
    logger.info("ASSISTMENTS: Downloading student interaction data (Axis 2 fallback)...")
    logger.info("=" * 60)

    assist_raw = RAW_DIR / "assistments"
    assist_out = PROCESSED_DIR / "assistments"
    assist_raw.mkdir(parents=True, exist_ok=True)
    assist_out.mkdir(parents=True, exist_ok=True)

    csv_path = assist_raw / "data.csv"

    if not csv_path.exists():
        try:
            import requests
            logger.info("  Downloading ASSISTments 2009 skill-builder data...")
            resp = requests.get(ASSISTMENTS_DIRECT_URL, timeout=60)
            resp.raise_for_status()
            csv_path.write_text(resp.text)
            logger.info(f"  Downloaded {len(resp.text)} bytes")
        except Exception as e:
            logger.error(f"  Download failed: {e}")
            logger.info("  Creating simulated ASSISTments data instead")
            _create_simulated_assistments(assist_out)
            return True
    else:
        logger.info("  ASSISTments data already downloaded")

    interactions = []
    skills = set()
    students = set()
    row_count = 0

    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            if max_rows and row_count > max_rows:
                break

            uid   = row.get("user_id",    row.get("student_id", "")).strip()
            skill = row.get("skill_name", row.get("skill_id",
                    row.get("skill", ""))).strip()
            correct = row.get("correct", "0").strip()

            if not uid or not skill:
                continue

            students.add(uid)
            skills.add(skill)

            interactions.append({
                "user_id":       uid,
                "skill_name":    skill,
                "correct":       correct,
                "hint_count":    row.get("hint_count",    "0"),
                "attempt_count": row.get("attempt_count", "1"),
                "timestamp":     row.get("start_time",
                                 row.get("timestamp", "")),
                "mrl_divergence": 0.0,   # Filled by inject_mrl step if enabled
            })

    fieldnames = ["user_id", "skill_name", "correct", "hint_count",
                  "attempt_count", "timestamp", "mrl_divergence"]

    # Strategy C: retroactively inject MRL divergence from skill names
    if inject_mrl and interactions:
        interactions = _inject_mrl_into_assistments(interactions, assist_out)

    with open(assist_out / "interactions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(interactions)

    skill_mapping = {
        skill: f"assist_{hashlib.md5(skill.encode()).hexdigest()[:10]}"
        for skill in sorted(skills)
    }
    with open(assist_out / "skill_mapping.json", "w") as f:
        json.dump(skill_mapping, f, indent=2)

    mrl_coverage = sum(1 for i in interactions if i.get("mrl_divergence", 0.0) != 0.0)
    logger.info(
        f"✓ ASSISTments: {len(interactions)} interactions, "
        f"{len(students)} students, {len(skills)} skills, "
        f"MRL coverage: {mrl_coverage}/{len(interactions)}"
    )
    return True


def _inject_mrl_into_assistments(
    interactions: List[Dict],
    assist_out: Path,
) -> List[Dict]:
    """
    Strategy C: Inject real MRL divergence into ASSISTments interactions.

    Maps skill_name → query template → MRL divergence vs corpus chunks.
    Requires sentence-transformers. Falls back to heuristic if unavailable.
    """
    logger.info("  Strategy C: Injecting MRL divergence into ASSISTments...")

    # Load corpus chunks for divergence computation
    corpus_chunks = []
    for corpus_path in [
        PROCESSED_DIR / "openstax_full" / "chunks.json",
        PROCESSED_DIR / "physics_v1"   / "chunks.json",
    ]:
        if corpus_path.exists():
            with open(corpus_path) as f:
                corpus_chunks = json.load(f)
            logger.info(f"  Using corpus: {corpus_path} ({len(corpus_chunks)} chunks)")
            break

    if not corpus_chunks:
        logger.warning("  No corpus found for MRL injection — using heuristic fallback")

    try:
        from models.mrl_encoder import MRLEncoder
        from data.pipelines.concept_alignment import ConceptAligner

        encoder = MRLEncoder({
            "model_name":           "nomic-ai/nomic-embed-text-v1.5",
            "matryoshka_dims":      [64, 128, 256, 512, 768],
            "full_dim":             768,
            "normalize_embeddings": True,
            "trust_remote_code":    True,
        })

        if not encoder._model_loaded:
            raise ImportError("sentence-transformers not installed")

        aligner = ConceptAligner(encoder=encoder)
        updated = aligner.inject_mrl_into_assistments(interactions, corpus_chunks)
        logger.info("  ✓ Real MRL divergence injected via ConceptAligner")
        return updated

    except (ImportError, Exception) as e:
        logger.warning(f"  MRL injection failed ({e}), using heuristic proxy")
        from data.pipelines.concept_alignment import ConceptAligner
        aligner = ConceptAligner(encoder=None)
        return aligner._heuristic_mrl_injection(interactions)


def _create_simulated_assistments(assist_out: Path):
    """Create simulated ASSISTments data for testing."""
    import numpy as np
    rng = np.random.RandomState(42)

    skills = [
        "Addition", "Subtraction", "Multiplication", "Division",
        "Fractions", "Decimals", "Percentages", "Ratios",
        "Linear Equations", "Quadratic Equations", "Geometry Basics",
        "Area and Perimeter", "Pythagorean Theorem", "Probability",
        "Statistics Mean", "Exponents", "Order of Operations",
        "Negative Numbers", "Absolute Value", "Scientific Notation"
    ]
    skill_mapping = {s: f"assist_sim_{i}" for i, s in enumerate(skills)}

    interactions = []
    for student in range(200):
        ability = rng.uniform(0.3, 0.9)
        n_ints  = rng.randint(20, 100)
        for j in range(n_ints):
            skill   = rng.choice(skills)
            correct = "1" if rng.random() < ability else "0"
            # Heuristic MRL divergence: anti-correlates with correctness
            mrl = round(float(rng.beta(2, 5)) * (0.15 if correct == "0" else 0.05), 4)
            interactions.append({
                "user_id":       f"sim_student_{student}",
                "skill_name":    skill,
                "correct":       correct,
                "hint_count":    str(rng.randint(0, 3)),
                "attempt_count": str(rng.randint(1, 4)),
                "timestamp":     "",
                "mrl_divergence": mrl,
            })

    fieldnames = ["user_id", "skill_name", "correct", "hint_count",
                  "attempt_count", "timestamp", "mrl_divergence"]
    with open(assist_out / "interactions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(interactions)

    with open(assist_out / "skill_mapping.json", "w") as f:
        json.dump(skill_mapping, f, indent=2)

    logger.info(f"  ✓ Simulated ASSISTments: {len(interactions)} interactions "
                f"(with heuristic MRL divergence)")


# ═════════════════════════════════════════════════════════════════════════════
# 4. EdNet  — KT / forgetting evaluation (Axis 2, PRIMARY)
# ═════════════════════════════════════════════════════════════════════════════

def prepare_ednet(
    max_students: Optional[int] = None,
    simulate: bool = False,
    compute_mrl: bool = False,
):
    """
    Process EdNet dataset for KT/forgetting evaluation (Axis 2 PRIMARY).

    EdNet is PREFERRED over ASSISTments because:
      - Real timestamps → true Ebbinghaus forgetting evaluation
      - Concept tags → concept-level BKT without proxy mapping
      - Question text (KT4) → real MRL divergence computation
      - Scale: 131M interactions, 784K students

    Download:
      wget https://github.com/riiid/ednet/raw/master/data/KT1/EdNet-KT1.zip
      unzip EdNet-KT1.zip -d data/raw/ednet/
      wget https://github.com/riiid/ednet/raw/master/data/contents/questions.csv
      cp questions.csv data/raw/ednet/

    Args:
        max_students: Cap on students to process (None = all).
        simulate:     Use simulated data (no download required).
        compute_mrl:  Compute real MRL from question text (needs KT4 + sentence-transformers).
    """
    logger.info("=" * 60)
    logger.info("EDNET: Processing student interaction data (Axis 2 PRIMARY)...")
    logger.info("=" * 60)

    from data.pipelines.ednet_pipeline import EdNetPipeline

    pipeline = EdNetPipeline(
        raw_dir=str(RAW_DIR / "ednet"),
        processed_dir=str(PROCESSED_DIR),
        kt_subset="KT1",
    )

    if simulate:
        pipeline._create_simulated_questions()
        interactions = pipeline._create_simulated_interactions(
            max_students=max_students or 200,
            min_interactions=10,
        )
        pipeline._save(interactions)
        logger.info("✓ EdNet: simulated data saved (use --ednet-real for real data)")
        return True

    ok = pipeline.process(
        max_students=max_students,
        min_interactions=10,
        compute_mrl=compute_mrl,
    )

    if ok:
        stats_path = PROCESSED_DIR / "ednet" / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            logger.info(
                f"✓ EdNet: {stats['n_students']} students, "
                f"{stats['n_interactions']} interactions, "
                f"MRL coverage={stats['mrl_coverage']:.1%}"
            )
    else:
        logger.warning("  EdNet real data not found. Using simulated fallback.")
        pipeline._create_simulated_questions()
        interactions = pipeline._create_simulated_interactions(
            max_students=max_students or 200,
        )
        pipeline._save(interactions)

    return True


# ═════════════════════════════════════════════════════════════════════════════
# 5. Junyi Academy  — supplementary temporal interactions
# ═════════════════════════════════════════════════════════════════════════════

def prepare_junyi():
    """Set up Junyi Academy data (requires Kaggle CLI or manual download)."""
    logger.info("=" * 60)
    logger.info("JUNYI: Setting up Junyi Academy data...")
    logger.info("=" * 60)

    junyi_out = PROCESSED_DIR / "junyi"
    junyi_out.mkdir(parents=True, exist_ok=True)

    concepts_path = junyi_out / "concepts.json"
    if concepts_path.exists():
        logger.info("  Junyi already processed")
        return True

    logger.info("  Junyi requires manual download from Kaggle.")
    logger.info("  Creating simulated Junyi data for testing...")
    _create_simulated_junyi(junyi_out)
    return True


def _create_simulated_junyi(junyi_out: Path):
    """Create simulated Junyi data."""
    import numpy as np
    rng = np.random.RandomState(99)

    subjects = ["algebra", "geometry", "arithmetic", "statistics", "trigonometry"]
    concepts = []
    for s_idx, subject in enumerate(subjects):
        for c_idx in range(10):
            name = f"{subject}_concept_{c_idx}"
            concepts.append({
                "concept_id":    f"junyi_{s_idx}_{c_idx:02d}",
                "name":          name,
                "subject":       subject,
                "depth_level":   c_idx % 3,
                "chapter_order": s_idx * 100 + c_idx,
            })

    with open(junyi_out / "concepts.json", "w") as f:
        json.dump(concepts, f, indent=2)

    interactions = []
    for student in range(100):
        for _ in range(rng.randint(10, 50)):
            concept = rng.choice(concepts)
            interactions.append({
                "user_id":    f"junyi_student_{student}",
                "concept_id": concept["concept_id"],
                "correct":    int(rng.random() > 0.4),
                "timestamp":  int(1_600_000_000 + rng.randint(0, 10_000_000)),
            })

    with open(junyi_out / "interactions_summary.json", "w") as f:
        json.dump(interactions, f)

    logger.info(f"  ✓ Simulated Junyi: {len(concepts)} concepts, "
                f"{len(interactions)} interactions")


# ═════════════════════════════════════════════════════════════════════════════
# 6. MOOCCube  — End-to-end retrieval evaluation (Axis 3)
# ═════════════════════════════════════════════════════════════════════════════

def prepare_mooccube():
    """
    Set up MOOCCube dataset (Axis 3: end-to-end retrieval evaluation).

    MOOCCube is the ONLY dataset that provides both a prereq graph AND
    student logs in the same domain — enabling end-to-end evaluation where
    retrieval quality is measured against real held-out student outcomes.

    Download: https://github.com/THU-KEG/MOOCCube
    """
    logger.info("=" * 60)
    logger.info("MOOCCUBE: Setting up dataset (Axis 3: end-to-end evaluation)...")
    logger.info("=" * 60)

    mc_raw = RAW_DIR / "mooccube"
    mc_out = PROCESSED_DIR / "mooccube"
    mc_out.mkdir(parents=True, exist_ok=True)

    if (mc_out / "concepts.json").exists():
        logger.info("  MOOCCube already processed")
        return True

    # Check for real data
    if (mc_raw / "entities" / "concepts.json").exists():
        logger.info("  Real MOOCCube data found — processing...")
        return _process_real_mooccube(mc_raw, mc_out)
    else:
        logger.info("  MOOCCube not downloaded. Creating simulated data.")
        logger.info("  For real data: https://github.com/THU-KEG/MOOCCube")
        _create_simulated_mooccube(mc_out)
        return True


def _process_real_mooccube(mc_raw: Path, mc_out: Path) -> bool:
    """Process real MOOCCube data if available."""
    try:
        with open(mc_raw / "entities" / "concepts.json") as f:
            raw_concepts = json.load(f)

        concepts = []
        for c in raw_concepts:
            concepts.append({
                "concept_id":    f"mc_{c['concept_id']}",
                "name":          c.get("name", c["concept_id"]),
                "description":   c.get("description", ""),
                "depth_level":   1,
                "chapter_order": 0,
                "subject":       "mooc",
                "dataset":       "mooccube",
            })

        with open(mc_out / "concepts.json", "w") as f:
            json.dump(concepts, f, indent=2)

        # Process prereq edges
        edges = []
        relations_dir = mc_raw / "relations" / "concept-concept"
        if relations_dir.exists():
            for rel_file in relations_dir.glob("*.json"):
                with open(rel_file) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            rel = json.loads(line)
                            edges.append({
                                "source_id":  f"mc_{rel['source']}",
                                "target_id":  f"mc_{rel['target']}",
                                "confidence": float(rel.get("score", 0.7)),
                                "source":     "mooccube_human",
                            })
                        except (json.JSONDecodeError, KeyError):
                            continue

        with open(mc_out / "edges.json", "w") as f:
            json.dump(edges, f, indent=2)

        logger.info(
            f"✓ MOOCCube: {len(concepts)} concepts, {len(edges)} prereq edges"
        )
        return True

    except Exception as e:
        logger.error(f"  MOOCCube processing error: {e}")
        _create_simulated_mooccube(mc_out)
        return True


def _create_simulated_mooccube(mc_out: Path):
    """Create simulated MOOCCube data with realistic structure."""
    import numpy as np
    rng = np.random.RandomState(77)

    courses = [
        ("intro_programming",  ["variables", "loops", "functions", "classes"]),
        ("data_structures",    ["arrays", "linked_lists", "trees", "graphs"]),
        ("machine_learning",   ["linear_regression", "gradient_descent",
                                "neural_networks", "backpropagation"]),
        ("calculus",           ["limits", "derivatives", "integrals", "series"]),
        ("statistics",         ["probability", "distributions", "hypothesis_testing",
                                "regression"]),
    ]

    concepts = []
    edges    = []
    chapter  = 0

    for course_name, topics in courses:
        prev_cid = None
        for t_idx, topic in enumerate(topics):
            cid = f"mc_{course_name}_{topic}"
            concepts.append({
                "concept_id":    cid,
                "name":          topic.replace("_", " ").title(),
                "description":   f"Concept: {topic} in {course_name}",
                "depth_level":   t_idx % 3,
                "chapter_order": chapter,
                "subject":       course_name,
                "dataset":       "mooccube",
            })
            if prev_cid:
                edges.append({
                    "source_id":  prev_cid,
                    "target_id":  cid,
                    "confidence": 0.85,
                    "source":     "mooccube_simulated",
                })
            prev_cid = cid
            chapter += 10

    with open(mc_out / "concepts.json", "w") as f:
        json.dump(concepts, f, indent=2)
    with open(mc_out / "edges.json", "w") as f:
        json.dump(edges, f, indent=2)

    # Simulated student logs (watch_ratio as engagement proxy)
    interactions = []
    student_ids = [f"mc_student_{i}" for i in range(150)]
    ts_base = 1_600_000_000

    for sid in student_ids:
        ability = rng.uniform(0.4, 0.95)
        for _ in range(rng.randint(10, 60)):
            concept = rng.choice(concepts)
            watch_ratio = float(np.clip(
                rng.normal(ability, 0.15), 0.0, 1.0
            ))
            interactions.append({
                "user_id":        sid,
                "concept_id":     concept["concept_id"],
                "watch_ratio":    round(watch_ratio, 3),
                "correct":        int(watch_ratio > 0.8),
                "timestamp":      int(ts_base + rng.randint(0, 5_000_000)),
                "mrl_divergence": round(float(rng.beta(2, 5)) * (1 - watch_ratio) * 0.3, 4),
            })

    with open(mc_out / "student_logs.json", "w") as f:
        json.dump(interactions, f)

    logger.info(
        f"  ✓ Simulated MOOCCube: {len(concepts)} concepts, "
        f"{len(edges)} prereq edges, {len(interactions)} student interactions"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Master preparation function
# ═════════════════════════════════════════════════════════════════════════════

def prepare_all(
    mode: str = "quick",
    only: Optional[str] = None,
    ednet_simulate: bool = False,
    inject_mrl: bool = False,
) -> Dict[str, bool]:
    """
    Prepare all datasets for the three evaluation axes.

    Evaluation axis mapping:
      Axis 1 (Prereq Graph):      openstax + lecturebank
      Axis 2 (KT/Forgetting):     ednet (primary) + assistments (fallback)
      Axis 3 (End-to-End):        mooccube
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    if only:
        targets = [only]
    else:
        # All datasets in recommended order
        targets = ["openstax", "lecturebank", "ednet", "assistments", "junyi", "mooccube"]

    for target in targets:
        if target == "openstax":
            max_ch       = 5 if mode == "quick" else None
            multi_subj   = (mode == "full")
            results["openstax"] = prepare_openstax(
                max_chapters=max_ch, multi_subject=multi_subj
            )
        elif target == "lecturebank":
            results["lecturebank"] = prepare_lecturebank()
        elif target == "ednet":
            max_s = 200 if mode == "quick" else None
            results["ednet"] = prepare_ednet(
                max_students=max_s,
                simulate=ednet_simulate,
                compute_mrl=False,   # Requires KT4 + sentence-transformers; opt-in
            )
        elif target == "assistments":
            max_r = 50000 if mode == "quick" else None
            results["assistments"] = prepare_assistments(
                max_rows=max_r,
                inject_mrl=inject_mrl,
            )
        elif target == "junyi":
            results["junyi"] = prepare_junyi()
        elif target == "mooccube":
            results["mooccube"] = prepare_mooccube()

    # Print summary
    print("\n" + "=" * 65)
    print("DATA PREPARATION SUMMARY — PLEDGE-KARMA THREE-AXIS EVALUATION")
    print("=" * 65)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    print("\n  Evaluation Axis → Dataset Mapping:")
    print("  Axis 1 (Prereq Graph):   OpenStax (multi-subject) + LectureBank")
    print("  Axis 2 (KT/Forgetting):  EdNet [primary] + ASSISTments [fallback]")
    print("  Axis 3 (End-to-End):     MOOCCube")

    from data.data_loader import DataLoader
    loader    = DataLoader(str(PROCESSED_DIR))
    available = loader.get_available_datasets()
    print(f"\n  Available datasets: {available}")
    print(f"  Data directory:     {PROCESSED_DIR.absolute()}")
    print("=" * 65)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="PLEDGE-KARMA Data Preparation (Three-Axis Strategy)"
    )
    parser.add_argument(
        "--mode", choices=["quick", "full"], default="quick",
        help="quick: physics only + 50K rows; full: multi-subject + all data"
    )
    parser.add_argument(
        "--only",
        choices=["openstax", "lecturebank", "assistments", "ednet",
                 "junyi", "mooccube"],
        help="Process only this dataset"
    )
    parser.add_argument(
        "--ednet-simulate", action="store_true",
        help="Use simulated EdNet data (no download required)"
    )
    parser.add_argument(
        "--inject-mrl", action="store_true",
        help="Inject real MRL divergence into ASSISTments via ConceptAligner "
             "(Strategy C; requires sentence-transformers)"
    )
    args = parser.parse_args()

    try:
        import requests
    except ImportError:
        logger.info("Installing requests for HTTP downloads...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "requests"],
            capture_output=True
        )

    prepare_all(
        mode=args.mode,
        only=args.only,
        ednet_simulate=args.ednet_simulate,
        inject_mrl=args.inject_mrl,
    )