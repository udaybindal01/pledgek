"""
validate_mrl_divergence.py — Empirically validate the MRL divergence signal.

The core claim in the paper: high MRL divergence on a student's query 
(sim_768D >> sim_64D) predicts they are overconfident — they use the right 
vocabulary but don't understand the concept deeply.

This script validates that claim using ASSISTments data + real MRL embeddings:

    Experiment:
        1. For each student interaction, compute MRL divergence of their
           query against the retrieved chunk.
        2. Split interactions into high-divergence and low-divergence groups.
        3. Compare subsequent correctness rates between groups.
        4. If MRL divergence is a valid metacognitive signal:
               high_divergence_group.accuracy < low_divergence_group.accuracy
           because high divergence → student is reaching beyond their knowledge.

    Additionally:
        - Compute point-biserial correlation between divergence and next_correct
        - Plot divergence distribution by outcome (for paper Figure)

Run AFTER installing sentence-transformers:
    pip install sentence-transformers
    python scripts/validate_mrl_divergence.py \
        --assistments data/raw/assistments.csv \
        --output outputs/mrl_validation/

The output table goes directly into the paper as Table X (MRL Signal Validation).
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_assistments_with_queries(path: str) -> pd.DataFrame:
    """
    Load ASSISTments and synthesize natural-language queries from skill names.
    
    Since ASSISTments records skill names (not free-text queries), we
    generate queries using templates — the same mechanism KARMA would
    use in production.
    
    e.g. "Solving Linear Equations" → "How do I solve linear equations?"
    """
    df = pd.read_csv(path, low_memory=False)
    
    # Normalize columns
    col_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if "user" in lc or "student" in lc:
            col_map[col] = "user_id"
        elif "skill" in lc or "kc" in lc or "concept" in lc:
            col_map[col] = "skill_name"
        elif lc == "correct":
            col_map[col] = "correct"
        elif "order" in lc or "time" in lc:
            col_map[col] = "order_id"
    df = df.rename(columns=col_map)
    
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["skill_name"])
    
    if "order_id" in df.columns:
        df = df.sort_values(["user_id", "order_id"])
    
    # Generate natural-language queries from skill names
    def skill_to_query(skill: str) -> str:
        skill = str(skill).strip()
        # Simple templates covering most ASSISTments skill patterns
        if any(w in skill.lower() for w in ["solving", "find", "calculate", "compute"]):
            return f"How do I solve problems involving {skill.lower()}?"
        elif any(w in skill.lower() for w in ["definition", "what is", "meaning"]):
            return f"What is {skill.lower()}?"
        elif any(w in skill.lower() for w in ["theorem", "law", "rule", "property"]):
            return f"Explain the {skill.lower()}"
        else:
            return f"Help me understand {skill.lower()}"
    
    df["query"] = df["skill_name"].apply(skill_to_query)
    
    logger.info(f"Loaded {len(df)} interactions, "
                f"{df['user_id'].nunique()} students, "
                f"{df['skill_name'].nunique()} skills")
    return df


def compute_mrl_divergences(
    queries: List[str],
    corpus_chunks: List[str],
    encoder,
    batch_size: int = 64
) -> np.ndarray:
    """
    Compute MRL divergence for a list of (query, best-matching-chunk) pairs.
    
    For each query, we find the top-1 chunk match at 768D (what standard RAG
    would return), then compute the MRL divergence between query and that chunk.
    
    Returns: array of divergence scores, one per query
    """
    logger.info(f"Encoding {len(queries)} queries and {len(corpus_chunks)} chunks...")
    
    query_embs  = encoder.encode(queries,        prompt_name="search_query",    show_progress=True)
    corpus_embs = encoder.encode(corpus_chunks,  prompt_name="search_document", show_progress=True)
    
    # Build corpus matrix at 64D and 768D
    corpus_64d  = np.stack([e.at_dim(64)  for e in corpus_embs])
    corpus_768d = np.stack([e.at_dim(768) for e in corpus_embs])
    
    divergences = []
    for q_emb in query_embs:
        q_64d  = q_emb.at_dim(64).reshape(1, -1)
        q_768d = q_emb.at_dim(768).reshape(1, -1)
        
        # Find top-1 match at 768D (what the student would be shown)
        sims_768d = (q_768d @ corpus_768d.T).flatten()
        top_idx   = int(np.argmax(sims_768d))
        
        # Compute divergence: sim_768D - sim_64D for (query, best_chunk)
        sim_fine   = float(sims_768d[top_idx])
        sim_coarse = float((q_64d @ corpus_64d[top_idx:top_idx+1].T).flatten()[0])
        
        divergences.append(sim_fine - sim_coarse)
    
    return np.array(divergences)


def validate_divergence_predicts_errors(
    df: pd.DataFrame,
    divergences: np.ndarray,
    output_dir: Path
) -> Dict:
    """
    Core validation: does high MRL divergence predict subsequent errors?
    
    Protocol:
        1. For each student interaction t, we have divergence(t).
        2. "next_correct" = correctness at interaction t+1 on the SAME skill.
        3. We test: high_divergence → lower next_correct.
    """
    assert len(divergences) == len(df), \
        f"Mismatch: {len(divergences)} divergences vs {len(df)} rows"
    
    df = df.copy()
    df["mrl_divergence"] = divergences
    
    # Compute next_correct per (user, skill)
    df = df.sort_values(["user_id", "skill_name", "order_id"] 
                         if "order_id" in df.columns 
                         else ["user_id", "skill_name"])
    df["next_correct"] = df.groupby(["user_id", "skill_name"])["correct"].shift(-1)
    df = df.dropna(subset=["next_correct"])
    df["next_correct"] = df["next_correct"].astype(int)
    
    # Split into high/low divergence groups at median
    median_div = df["mrl_divergence"].median()
    high_div = df[df["mrl_divergence"] > median_div]
    low_div  = df[df["mrl_divergence"] <= median_div]
    
    high_acc = float(high_div["next_correct"].mean())
    low_acc  = float(low_div["next_correct"].mean())
    
    # Point-biserial correlation (divergence → binary outcome)
    from scipy import stats
    corr, pval = stats.pointbiserialr(df["mrl_divergence"], df["next_correct"])
    
    # Quartile breakdown
    df["quartile"] = pd.qcut(df["mrl_divergence"], q=4, labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])
    quartile_acc = df.groupby("quartile")["next_correct"].mean()
    
    results = {
        "n_samples": len(df),
        "median_divergence": round(float(median_div), 4),
        "high_divergence_accuracy": round(high_acc, 4),
        "low_divergence_accuracy":  round(low_acc,  4),
        "accuracy_gap": round(low_acc - high_acc, 4),
        "point_biserial_correlation": round(float(corr), 4),
        "p_value": round(float(pval), 6),
        "significant": pval < 0.05,
        "quartile_accuracy": {str(q): round(float(a), 4) 
                              for q, a in quartile_acc.items()},
        "interpretation": (
            "VALIDATED: High MRL divergence predicts lower next-question accuracy. "
            f"Gap = {low_acc - high_acc:.3f}, r = {corr:.3f}, p = {pval:.4f}"
            if (low_acc > high_acc and pval < 0.05) else
            "NOT VALIDATED: No significant relationship found. "
            "Consider: (1) real MRL embeddings required, (2) larger sample needed."
        )
    }
    
    # Print summary table for paper
    print("\n=== MRL Divergence Validation Results ===")
    print(f"N = {results['n_samples']} interactions")
    print(f"\nNext-question accuracy by divergence group:")
    print(f"  Low  divergence (Q1-Q2): {low_acc:.3f}")
    print(f"  High divergence (Q3-Q4): {high_acc:.3f}")
    print(f"  Gap:                     {low_acc - high_acc:+.3f}")
    print(f"\nPoint-biserial correlation: r = {corr:.3f}, p = {pval:.4f}")
    print(f"\nQuartile breakdown:")
    for q, a in quartile_acc.items():
        print(f"  {q}: {a:.3f}")
    print(f"\n{'✓ VALIDATED' if results['significant'] else '✗ NOT VALIDATED'}: "
          f"{results['interpretation']}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "mrl_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Try to produce a plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot 1: accuracy by quartile
        qs = [str(q) for q in quartile_acc.index]
        accs = [float(a) for a in quartile_acc.values]
        axes[0].bar(qs, accs, color=["#4CAF50", "#8BC34A", "#FF9800", "#F44336"])
        axes[0].set_xlabel("MRL Divergence Quartile")
        axes[0].set_ylabel("Next-Question Accuracy")
        axes[0].set_title("MRL Divergence vs Learning Outcome")
        axes[0].set_ylim(0, 1)
        axes[0].axhline(df["next_correct"].mean(), color="black", 
                        linestyle="--", alpha=0.5, label="Overall mean")
        axes[0].legend()
        
        # Plot 2: divergence distribution by outcome
        correct_div   = df[df["next_correct"] == 1]["mrl_divergence"]
        incorrect_div = df[df["next_correct"] == 0]["mrl_divergence"]
        axes[1].hist(correct_div,   bins=30, alpha=0.6, label="Next: Correct",   color="#4CAF50", density=True)
        axes[1].hist(incorrect_div, bins=30, alpha=0.6, label="Next: Incorrect", color="#F44336", density=True)
        axes[1].set_xlabel("MRL Divergence (sim_768D - sim_64D)")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Divergence Distribution by Next-Question Outcome")
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "mrl_divergence_validation.pdf", bbox_inches="tight")
        plt.savefig(output_dir / "mrl_divergence_validation.png", dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {output_dir}/mrl_divergence_validation.pdf")
    except ImportError:
        print("matplotlib not available — skipping plot generation")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistments", default="data/raw/assistments.csv")
    parser.add_argument("--corpus",      default="data/processed/math_chunks.json",
                        help="JSON file with list of {'text': ...} objects")
    parser.add_argument("--output",      default="outputs/mrl_validation/")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Sample N interactions for speed (use all for final paper)")
    args = parser.parse_args()

    # Load encoder (requires sentence-transformers)
    from models.mrl_encoder import MRLEncoder
    encoder = MRLEncoder({
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "matryoshka_dims": [64, 128, 256, 512, 768],
        "full_dim": 768,
        "batch_size": 64,
        "device": "cpu",
        "normalize_embeddings": True,
        "trust_remote_code": True
    })

    if not encoder._model_loaded:
        print("ERROR: sentence-transformers not installed.")
        print("Run: pip install sentence-transformers")
        print("This experiment requires real MRL embeddings to be meaningful.")
        sys.exit(1)

    # Load data
    df = load_assistments_with_queries(args.assistments)
    
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {args.max_samples} interactions for speed")

    # Load corpus chunks
    corpus_path = Path(args.corpus)
    if corpus_path.exists():
        with open(corpus_path) as f:
            corpus_data = json.load(f)
        corpus_chunks = [c["text"] for c in corpus_data if "text" in c][:2000]
    else:
        logger.warning(f"Corpus not found at {corpus_path}. "
                       f"Using skill names as proxy corpus.")
        corpus_chunks = df["skill_name"].unique().tolist()
    
    logger.info(f"Corpus size: {len(corpus_chunks)} chunks")

    # Compute divergences
    queries = df["query"].tolist()
    divergences = compute_mrl_divergences(queries, corpus_chunks, encoder)

    # Validate
    output_dir = Path(args.output)
    results = validate_divergence_predicts_errors(df, divergences, output_dir)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()