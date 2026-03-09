#!/usr/bin/env python3
"""
PLEDGE-KARMA Data Preparation Script
=======================================
Downloads and processes all datasets needed for the research.

Usage:
    # Quick mode: first 5 OpenStax chapters + first 50K ASSISTments rows
    python data/prepare_data.py --mode quick

    # Full mode: all available data
    python data/prepare_data.py --mode full

    # Single dataset
    python data/prepare_data.py --only openstax
    python data/prepare_data.py --only lecturebank
    python data/prepare_data.py --only assistments
    python data/prepare_data.py --only junyi

Datasets:
    OpenStax Physics  — textbook corpus (free API, CC-BY)
    LectureBank       — prerequisite graph ground truth (GitHub)
    ASSISTments 2012  — student interaction logs (public CSV)
    Junyi Academy     — temporal interactions (Kaggle)
    MOOCCube          — prereq graph + student logs (simulated fallback)
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

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ═════════════════════════════════════════════════════════════════════════════
# 1. OpenStax
# ═════════════════════════════════════════════════════════════════════════════
def prepare_openstax(max_chapters: Optional[int] = None):
    """Download and process OpenStax dataset from HuggingFace."""
    logger.info("=" * 60)
    logger.info("OPENSTAX: Downloading HuggingFaceTB/openstax_paragraphs...")
    logger.info("=" * 60)

    from data.pipelines.hf_openstax_pipeline import HFOpenStaxPipeline

    pipeline = HFOpenStaxPipeline(output_dir=str(PROCESSED_DIR))

    try:
        # Quick mode limits to 1 book, Full mode processes the entire corpus
        max_books = 1 if max_chapters is not None else None
        book_filter = "physics" if max_chapters is not None else ""
        concepts, chunks = pipeline.process_dataset(max_books=max_books, book_name_filter=book_filter)
        pipeline.save_processed("openstax_full")
        logger.info(f"✓ OpenStax (HF): {len(concepts)} concepts, {len(chunks)} chunks")
        return True
    except Exception as e:
        logger.error(f"  OpenStax download failed: {e}")
        logger.info("  Creating simulated OpenStax corpus for testing...")
        _create_simulated_openstax()
        return True


def _create_simulated_openstax():
    """Create simulated OpenStax corpus using the experiment runner's mock builder."""
    from experiments.run_experiment import build_mock_corpus
    concepts_obj, chunks_obj = build_mock_corpus(n_concepts=50, n_chunks=200)
    
    out_dir = PROCESSED_DIR / "openstax_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    concepts = []
    for c in concepts_obj:
        concepts.append({
            "concept_id": c.concept_id,
            "name": c.name,
            "description": c.description,
            "depth_level": c.depth_level,
            "chapter_order": c.chapter_order,
            "subject": c.subject,
            "tags": c.tags,
            "source_chunk_ids": c.source_chunk_ids,
        })
        
    chunks = []
    for c in chunks_obj:
        chunks.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "concept_ids": c.concept_ids,
            "prerequisite_concept_ids": c.prerequisite_concept_ids,
            "depth_level": c.depth_level,
            "chapter_order": c.chapter_order,
            "subject": c.subject,
            "source": c.source,
            "metadata": c.metadata,
        })
        
    with open(out_dir / "concepts.json", "w") as f:
        json.dump(concepts, f, indent=2)
    with open(out_dir / "chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
        
    logger.info(f"  ✓ Simulated OpenStax: {len(concepts)} concepts, {len(chunks)} chunks")


# ═════════════════════════════════════════════════════════════════════════════
# 2. LectureBank
# ═════════════════════════════════════════════════════════════════════════════
LECTUREBANK_REPO = "https://github.com/Yale-LILY/LectureBank.git"

def prepare_lecturebank():
    """Clone LectureBank repo and extract prerequisite labels."""
    logger.info("=" * 60)
    logger.info("LECTUREBANK: Cloning prerequisite graph data...")
    logger.info("=" * 60)

    lb_raw = RAW_DIR / "lecturebank"
    lb_out = PROCESSED_DIR / "lecturebank"
    lb_out.mkdir(parents=True, exist_ok=True)

    # Clone if not already present
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

    # Find and process prerequisite files
    prereq_edges = []
    concept_set = set()

    # LectureBank stores prereqs in various TSV files
    for tsv in lb_raw.rglob("*.tsv"):
        try:
            with open(tsv, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) >= 3:
                        src, tgt, label = row[0].strip(), row[1].strip(), row[2].strip()
                        if label in ("1", "True", "true"):
                            src_id = f"lb_{hashlib.md5(src.encode()).hexdigest()[:10]}"
                            tgt_id = f"lb_{hashlib.md5(tgt.encode()).hexdigest()[:10]}"
                            prereq_edges.append({
                                "source_id": src_id,
                                "target_id": tgt_id,
                                "source_name": src,
                                "target_name": tgt,
                                "confidence": 0.95,
                                "source": "lecturebank",
                            })
                            concept_set.add(src)
                            concept_set.add(tgt)
        except Exception:
            continue

    if not prereq_edges:
        logger.info("  No labeled TSV files found, using simulated data")
        _create_simulated_lecturebank(lb_out)
        return True

    # Save processed edges
    with open(lb_out / "prereq_edges.json", "w") as f:
        json.dump(prereq_edges, f, indent=2)

    # Save concept list
    concepts = [
        {"concept_id": f"lb_{hashlib.md5(c.encode()).hexdigest()[:10]}",
         "name": c, "source": "lecturebank"}
        for c in sorted(concept_set)
    ]
    with open(lb_out / "concepts.json", "w") as f:
        json.dump(concepts, f, indent=2)

    logger.info(f"✓ LectureBank: {len(prereq_edges)} edges, {len(concept_set)} concepts")
    return True


def _create_simulated_lecturebank(lb_out: Path):
    """Create simulated LectureBank data using the existing loader's simulate method."""
    from data.pipelines.prereq_graph_pipeline import LectureBankLoader

    loader = LectureBankLoader("__nonexistent__")
    dataset = loader.load()  # Falls back to simulated data

    edges = [e.to_dict() for e in dataset.positive_edges]
    with open(lb_out / "prereq_edges.json", "w") as f:
        json.dump(edges, f, indent=2)

    concepts = [
        {"concept_id": cid, "name": name, "source": "lb_simulated"}
        for name, cid in dataset.concept_map.items()
    ]
    with open(lb_out / "concepts.json", "w") as f:
        json.dump(concepts, f, indent=2)

    logger.info(f"  ✓ Simulated LectureBank: {len(edges)} edges, {len(concepts)} concepts")


# ═════════════════════════════════════════════════════════════════════════════
# 3. ASSISTments
# ═════════════════════════════════════════════════════════════════════════════
ASSISTMENTS_URL = (
    "https://sites.google.com/site/assistmaborern/"
    # The direct download URL for ASSISTments skill-builder data (2009-2010)
)
# Using the 2009-2010 skill-builder dataset which is smaller and easier to download
ASSISTMENTS_DIRECT_URL = (
    "https://raw.githubusercontent.com/hcnoh/knowledge-tracing-collection-tensorflow2.0/"
    "master/data/assistments09/data.csv"
)


def prepare_assistments(max_rows: Optional[int] = None):
    """Download and process ASSISTments dataset."""
    logger.info("=" * 60)
    logger.info("ASSISTMENTS: Downloading student interaction data...")
    logger.info("=" * 60)

    assist_raw = RAW_DIR / "assistments"
    assist_out = PROCESSED_DIR / "assistments"
    assist_raw.mkdir(parents=True, exist_ok=True)
    assist_out.mkdir(parents=True, exist_ok=True)

    csv_path = assist_raw / "data.csv"

    # Download if not present
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

    # Process the CSV
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

            # The KT collection format has columns:
            # user_id, skill_id/skill_name, correct, ...
            uid = row.get("user_id", row.get("student_id", "")).strip()
            skill = row.get("skill_name", row.get("skill_id",
                    row.get("skill", ""))).strip()
            correct = row.get("correct", "0").strip()

            if not uid or not skill:
                continue

            students.add(uid)
            skills.add(skill)

            interactions.append({
                "user_id": uid,
                "skill_name": skill,
                "correct": correct,
                "hint_count": row.get("hint_count", "0"),
                "attempt_count": row.get("attempt_count", "1"),
                "timestamp": row.get("start_time",
                             row.get("timestamp", "")),
            })

    # Save processed interactions
    fieldnames = ["user_id", "skill_name", "correct", "hint_count",
                  "attempt_count", "timestamp"]
    with open(assist_out / "interactions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in interactions:
            writer.writerow(row)

    # Save skill mapping
    skill_mapping = {
        skill: f"assist_{hashlib.md5(skill.encode()).hexdigest()[:10]}"
        for skill in sorted(skills)
    }
    with open(assist_out / "skill_mapping.json", "w") as f:
        json.dump(skill_mapping, f, indent=2)

    logger.info(
        f"✓ ASSISTments: {len(interactions)} interactions, "
        f"{len(students)} students, {len(skills)} skills"
    )
    return True


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
        n_ints = rng.randint(20, 100)
        for j in range(n_ints):
            skill = rng.choice(skills)
            correct = "1" if rng.random() < ability else "0"
            interactions.append({
                "user_id": f"sim_student_{student}",
                "skill_name": skill,
                "correct": correct,
                "hint_count": str(rng.randint(0, 3)),
                "attempt_count": str(rng.randint(1, 4)),
                "timestamp": "",
            })

    fieldnames = ["user_id", "skill_name", "correct", "hint_count",
                  "attempt_count", "timestamp"]
    with open(assist_out / "interactions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(interactions)

    with open(assist_out / "skill_mapping.json", "w") as f:
        json.dump(skill_mapping, f, indent=2)

    logger.info(f"  ✓ Simulated ASSISTments: {len(interactions)} interactions")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Junyi Academy
# ═════════════════════════════════════════════════════════════════════════════
def prepare_junyi():
    """Set up Junyi Academy data (requires Kaggle CLI or manual download)."""
    logger.info("=" * 60)
    logger.info("JUNYI ACADEMY: Setting up student interaction data...")
    logger.info("=" * 60)

    junyi_raw = RAW_DIR / "junyi"
    junyi_out = PROCESSED_DIR / "junyi"
    junyi_raw.mkdir(parents=True, exist_ok=True)
    junyi_out.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    if (junyi_raw / "Log_Problem.csv").exists():
        logger.info("  Junyi data already downloaded, processing...")
        from models.junyi_processor import JunyiProcessor
        proc = JunyiProcessor(str(junyi_raw), max_students=5000)
        proc.save_processed(str(PROCESSED_DIR))
        return True

    # Try Kaggle CLI
    kaggle_dataset = "junyiacademy/junyi-academy-online-learning-activity-dataset"
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_dataset,
             "-p", str(junyi_raw), "--unzip"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            logger.info("  Downloaded via Kaggle CLI")
            from models.junyi_processor import JunyiProcessor
            proc = JunyiProcessor(str(junyi_raw), max_students=5000)
            proc.save_processed(str(PROCESSED_DIR))
            return True
        else:
            logger.info(f"  Kaggle download failed: {result.stderr[:200]}")
    except FileNotFoundError:
        logger.info("  Kaggle CLI not found")
    except Exception as e:
        logger.info(f"  Kaggle download error: {e}")

    # Create simulated Junyi data
    logger.info("  Creating simulated Junyi data for testing...")
    _create_simulated_junyi(junyi_out)
    return True


def _create_simulated_junyi(junyi_out: Path):
    """Create simulated Junyi-like data for testing."""
    import numpy as np
    from datetime import datetime, timedelta

    rng = np.random.RandomState(42)

    # Math topics with prerequisite structure
    topics = [
        ("addition", 0), ("subtraction", 0), ("multiplication", 0),
        ("division", 0), ("fractions", 1), ("decimals", 1),
        ("ratios", 1), ("percentages", 1), ("linear_equations", 2),
        ("inequalities", 2), ("coordinate_geometry", 2),
        ("quadratic_equations", 2),
    ]

    concepts = []
    for i, (name, depth) in enumerate(topics):
        concepts.append({
            "concept_id": f"junyi_sim_{name}",
            "name": name.replace("_", " ").title(),
            "description": f"Junyi Academy topic: {name.replace('_', ' ')}",
            "depth_level": depth,
            "chapter_order": i * 10,
            "subject": "math",
            "tags": ["junyi", "simulated"],
        })

    with open(junyi_out / "concepts.json", "w") as f:
        json.dump(concepts, f, indent=2)

    # Simulate temporal interactions
    n_students = 500
    base_dt = datetime(2018, 8, 1)
    temporal_data = {
        "n_students": n_students,
        "n_interactions": 0,
        "retention_within_1d": 0.92,
        "retention_within_3d": 0.85,
        "retention_within_7d": 0.73,
        "retention_within_14d": 0.58,
        "retention_within_30d": 0.42,
        "median_gap_days": 2.3,
        "mean_gap_days": 5.1,
    }

    total_ints = 0
    for s in range(n_students):
        n = rng.randint(30, 150)
        total_ints += n
    temporal_data["n_interactions"] = total_ints

    with open(junyi_out / "interactions_summary.json", "w") as f:
        json.dump({"n_students": n_students, "n_interactions": total_ints}, f, indent=2)

    with open(junyi_out / "temporal_stats.json", "w") as f:
        json.dump(temporal_data, f, indent=2)

    logger.info(
        f"  ✓ Simulated Junyi: {len(concepts)} concepts, "
        f"{n_students} students, {total_ints} interactions"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 5. MOOCCube (simulated fallback)
# ═════════════════════════════════════════════════════════════════════════════
def prepare_mooccube():
    """Set up MOOCCube data (uses simulated fallback)."""
    logger.info("=" * 60)
    logger.info("MOOCCUBE: Setting up prerequisite graph data...")
    logger.info("=" * 60)

    mc_out = PROCESSED_DIR / "mooccube"
    mc_out.mkdir(parents=True, exist_ok=True)

    from data.pipelines.prereq_graph_pipeline import MOOCCubeLoader

    loader = MOOCCubeLoader(str(RAW_DIR / "mooccube"))
    concepts = loader.load_concepts()
    edges = loader.load_edges()

    # Save concepts
    with open(mc_out / "concepts.json", "w") as f:
        json.dump([{
            "concept_id": c.concept_id,
            "name": c.name,
            "description": c.description,
            "depth_level": c.depth_level,
            "chapter_order": c.chapter_order,
            "subject": c.subject,
            "tags": getattr(c, "tags", []),
        } for c in concepts], f, indent=2)

    # Save edges
    with open(mc_out / "prereq_edges.json", "w") as f:
        json.dump([e.to_dict() for e in edges], f, indent=2)

    logger.info(f"✓ MOOCCube: {len(concepts)} concepts, {len(edges)} edges")
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════════
def prepare_all(mode: str = "quick", only: Optional[str] = None):
    """Run all data preparation steps."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    if only:
        targets = [only]
    else:
        targets = ["openstax", "lecturebank", "assistments", "junyi", "mooccube"]

    for target in targets:
        if target == "openstax":
            max_ch = 5 if mode == "quick" else None
            results["openstax"] = prepare_openstax(max_chapters=max_ch)
        elif target == "lecturebank":
            results["lecturebank"] = prepare_lecturebank()
        elif target == "assistments":
            max_r = 50000 if mode == "quick" else None
            results["assistments"] = prepare_assistments(max_rows=max_r)
        elif target == "junyi":
            results["junyi"] = prepare_junyi()
        elif target == "mooccube":
            results["mooccube"] = prepare_mooccube()

    # Print summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    # Show what's available
    from data.data_loader import DataLoader
    loader = DataLoader(str(PROCESSED_DIR))
    available = loader.get_available_datasets()
    print(f"\nAvailable datasets: {available}")
    print(f"Data directory: {PROCESSED_DIR.absolute()}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="PLEDGE-KARMA Data Preparation"
    )
    parser.add_argument(
        "--mode", choices=["quick", "full"], default="quick",
        help="quick: first 5 chapters + 50K rows; full: everything"
    )
    parser.add_argument(
        "--only", choices=["openstax", "lecturebank", "assistments",
                           "junyi", "mooccube"],
        help="Process only this dataset"
    )
    args = parser.parse_args()

    # Install requests if needed (for ASSISTments download)
    try:
        import requests
    except ImportError:
        logger.info("Installing requests for HTTP downloads...")
        subprocess.run([sys.executable, "-m", "pip", "install", "requests"],
                       capture_output=True)

    prepare_all(mode=args.mode, only=args.only)
