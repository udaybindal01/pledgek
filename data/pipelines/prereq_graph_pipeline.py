"""
Prerequisite Graph Pipeline — LectureBank + MOOCCube
=====================================================
Builds and evaluates prerequisite edges from two human-annotated sources.

  LectureBank (Kann et al., ACL 2019)
    → Human-verified CS/NLP prereq pairs
    → Used as ground truth for automated extraction evaluation
    → Download: https://github.com/Yale-LILY/LectureBank

  MOOCCube (Yu et al., KDD 2020)
    → 700+ MOOC courses, concept graph + student logs
    → Download: https://github.com/THU-KEG/MOOCCube

The ablation table comparing MRL-only vs Linguistic-only vs Combined
is a direct paper contribution (Section 4 / Table 3).

Usage:
    # With real data
    python data/pipelines/prereq_graph_pipeline.py \
        --lecturebank data/raw/lecturebank \
        --mooccube   data/raw/mooccube \
        --corpus     data/processed/physics_v1/concepts.json \
        --output     data/processed/prereq_graph.json

    # Simulated data (for testing without downloads)
    python data/pipelines/prereq_graph_pipeline.py --simulate
"""

import re, json, logging, argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from sklearn.metrics import precision_score, recall_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_graph.graph_builder import ConceptNode
from models.mrl_encoder import MRLEncoder, MRLEmbedding
from data.processors.educational_processors import generate_id

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PrereqEdge:
    source_id:    str
    target_id:    str
    confidence:   float
    source:       str           # "lecturebank" | "mooccube" | "automated_*"
    human_label:  Optional[bool] = None

    def to_dict(self):
        return dict(source_id=self.source_id, target_id=self.target_id,
                    confidence=self.confidence, source=self.source,
                    human_label=self.human_label)


@dataclass
class AnnotationDataset:
    name:           str
    concept_map:    Dict[str, str]          # name → id
    positive_edges: List[PrereqEdge]
    negative_pairs: List[Tuple[str, str]]   # (id, id) confirmed non-prereqs


# ─────────────────────────────────────────────────────────────────────────────
# LectureBank loader
# ─────────────────────────────────────────────────────────────────────────────
class LectureBankLoader:
    """
    Loads LectureBank annotations.

    Expected files:
      concept_list.txt    — one concept name per line
      prereq_labels.tsv   — conceptA \\t conceptB \\t 1/0
    """

    def __init__(self, data_dir: str):
        self.dir = Path(data_dir)

    def load(self) -> AnnotationDataset:
        clist = self.dir / "concept_list.txt"
        pfile = self.dir / "prereq_labels.tsv"

        if not clist.exists() or not pfile.exists():
            logger.warning("LectureBank files not found — using simulated data.")
            return self._simulate()

        # concepts
        cmap: Dict[str, str] = {}
        with open(clist, encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    cmap[name] = generate_id(name.lower(), "lb")

        # labels
        pos_edges, neg_pairs = [], []
        with open(pfile, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                a, b, lab = parts[0], parts[1], parts[2].strip()
                if a not in cmap or b not in cmap:
                    continue
                if lab == "1":
                    pos_edges.append(PrereqEdge(
                        cmap[a], cmap[b], 0.95, "lecturebank", True))
                else:
                    neg_pairs.append((cmap[a], cmap[b]))

        logger.info(f"LectureBank: {len(cmap)} concepts, "
                    f"{len(pos_edges)} positive, {len(neg_pairs)} negative")
        return AnnotationDataset("lecturebank", cmap, pos_edges, neg_pairs)

    # ------------------------------------------------------------------
    def _simulate(self) -> AnnotationDataset:
        """
        Realistic CS/NLP concept hierarchy for testing.
        Covers two chains: algorithms and deep learning.
        """
        cmap = {
            "variables":           "lb_vars",
            "data types":          "lb_dtypes",
            "control flow":        "lb_control",
            "functions":           "lb_funcs",
            "recursion":           "lb_recursion",
            "arrays":              "lb_arrays",
            "linked lists":        "lb_ll",
            "trees":               "lb_trees",
            "graphs":              "lb_graphs",
            "sorting":             "lb_sort",
            "dynamic programming": "lb_dp",
            "gradient descent":    "lb_gd",
            "backpropagation":     "lb_bp",
            "neural networks":     "lb_nn",
            "word embeddings":     "lb_we",
            "attention":           "lb_attn",
            "transformers":        "lb_trans",
            "language models":     "lb_lm",
        }
        pos_pairs = [
            ("variables","data types"),("data types","control flow"),
            ("control flow","functions"),("functions","recursion"),
            ("arrays","sorting"),("arrays","linked lists"),
            ("linked lists","trees"),("trees","graphs"),
            ("sorting","dynamic programming"),
            ("gradient descent","backpropagation"),
            ("backpropagation","neural networks"),
            ("word embeddings","attention"),("attention","transformers"),
            ("transformers","language models"),
        ]
        neg_pairs_names = [
            ("recursion","transformers"),("sorting","attention"),
            ("variables","language models"),("linked lists","gradient descent"),
            ("arrays","transformers"),
        ]
        pos_edges = [
            PrereqEdge(cmap[a], cmap[b], 0.95, "lb_simulated", True)
            for a,b in pos_pairs if a in cmap and b in cmap
        ]
        neg_pairs = [
            (cmap[a], cmap[b])
            for a,b in neg_pairs_names if a in cmap and b in cmap
        ]
        return AnnotationDataset("lb_simulated", cmap, pos_edges, neg_pairs)


# ─────────────────────────────────────────────────────────────────────────────
# MOOCCube loader
# ─────────────────────────────────────────────────────────────────────────────
class MOOCCubeLoader:
    """
    Loads MOOCCube concepts + prerequisite relations.

    Expected files:
      entities/concepts.json         — list of {concept_id, name, description}
      relations/concept-concept/*.json — {source_concept, target_concept, score}
    """

    def __init__(self, data_dir: str):
        self.dir = Path(data_dir)

    def load_concepts(self) -> List[ConceptNode]:
        path = self.dir / "entities" / "concepts.json"
        if not path.exists():
            logger.warning("MOOCCube concepts.json not found — using simulated.")
            return self._sim_concepts()

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        concepts = []
        for i, item in enumerate(raw):
            name = item.get("name","")
            cid  = generate_id(str(item.get("concept_id", name)), "mc")
            concepts.append(ConceptNode(
                concept_id=cid, name=name,
                description=item.get("description", name),
                source_chunk_ids=[], depth_level=1,
                chapter_order=i, subject=item.get("field","general"),
                tags=["mooccube"],
            ))
        logger.info(f"MOOCCube: {len(concepts)} concepts")
        return concepts

    def load_edges(self) -> List[PrereqEdge]:
        rel_dir = self.dir / "relations" / "concept-concept"
        if not rel_dir.exists():
            logger.warning("MOOCCube relations not found — using simulated.")
            return self._sim_edges()

        edges = []
        for fp in rel_dir.glob("*.json"):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        src = generate_id(d.get("source_concept",""), "mc")
                        tgt = generate_id(d.get("target_concept",""), "mc")
                        sc  = float(d.get("score", d.get("confidence", 0.7)))
                        edges.append(PrereqEdge(src, tgt, sc, "mooccube", True))
                    except Exception:
                        continue
        logger.info(f"MOOCCube: {len(edges)} edges")
        return edges

    def load_student_logs(self, max_students: int = 2000) -> Dict[str,List[Dict]]:
        path = self.dir / "behaviors" / "watch.json"
        if not path.exists():
            logger.warning("MOOCCube watch.json not found — returning empty.")
            return {}
        logs: Dict[str, List[Dict]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d    = json.loads(line)
                    uid  = d.get("user_id","")
                    if not uid: continue
                    if len(logs) >= max_students and uid not in logs: continue
                    wr = float(d.get("watch_ratio", 0.5))
                    entry = dict(
                        student_id=uid,
                        concept_id=generate_id(d.get("concept_id",""),"mc"),
                        timestamp=d.get("timestamp",0),
                        response_quality=wr,
                        correct=wr > 0.8,
                    )
                    logs.setdefault(uid, []).append(entry)
                except Exception:
                    continue
        logger.info(f"MOOCCube: {len(logs)} students in behavior logs")
        return logs

    # ------------------------------------------------------------------
    def _sim_concepts(self) -> List[ConceptNode]:
        data = [
            ("mc_kinematics","Kinematics","Study of motion without forces",0,10),
            ("mc_dynamics","Dynamics","Study of forces and motion",1,20),
            ("mc_work","Work-Energy Theorem","W = ΔKE relationship",1,30),
            ("mc_momentum","Momentum","p = mv and conservation laws",1,40),
            ("mc_rotation","Rotational Motion","Angular kinematics and dynamics",2,50),
            ("mc_shm","Simple Harmonic Motion","Periodic motion and oscillation",2,60),
            ("mc_waves","Wave Motion","Mechanical waves and interference",2,70),
            ("mc_thermo","Thermodynamics","Heat, temperature, entropy",2,80),
        ]
        return [
            ConceptNode(concept_id=cid, name=name, description=desc,
                        source_chunk_ids=[], depth_level=depth,
                        chapter_order=order, subject="physics", tags=["mooccube"])
            for cid,name,desc,depth,order in data
        ]

    def _sim_edges(self) -> List[PrereqEdge]:
        pairs = [
            ("mc_kinematics","mc_dynamics",0.92),
            ("mc_dynamics","mc_work",0.88),
            ("mc_dynamics","mc_momentum",0.85),
            ("mc_work","mc_rotation",0.79),
            ("mc_rotation","mc_shm",0.81),
            ("mc_shm","mc_waves",0.86),
        ]
        return [PrereqEdge(s,t,c,"mc_simulated",True) for s,t,c in pairs]


# ─────────────────────────────────────────────────────────────────────────────
# Automated extractor using MRL
# ─────────────────────────────────────────────────────────────────────────────
PREREQ_LING_RE = re.compile(
    r"requires?\s+(?:knowledge\s+of\s+)?(.{5,50})|"
    r"builds?\s+on\s+(.{5,50})|"
    r"assumes?\s+(?:familiarity\s+with\s+)?(.{5,50})|"
    r"based\s+on\s+(.{5,50})|"
    r"extension\s+of\s+(.{5,50})",
    re.IGNORECASE,
)


class AutomatedExtractor:
    """
    Automated prerequisite edge extraction.
    Combines: chapter ordering + MRL cross-scale agreement + linguistic patterns.
    """

    def __init__(self, encoder: MRLEncoder, config: Dict):
        self.encoder         = encoder
        self.sim_thresh      = config.get("prerequisite_sim_threshold", 0.70)
        self.coarse_thresh   = config.get("cross_scale_agreement_threshold", 0.55)
        self.min_conf        = config.get("min_edge_confidence", 0.60)
        self.max_gap         = config.get("max_chapter_gap", 5)

    def extract(self,
                concepts: List[ConceptNode],
                use_mrl:        bool = True,
                use_linguistic: bool = True,
                use_ordering:   bool = True) -> List[PrereqEdge]:
        # Embed missing concepts
        to_embed = [c for c in concepts if c.embedding is None]
        if to_embed:
            texts = [f"{c.name}: {c.description}" for c in to_embed]
            embs  = self.encoder.encode_documents_batch(texts, show_progress=False)
            for c, e in zip(to_embed, embs):
                c.embedding = e.full_embedding

        sorted_c = sorted(concepts, key=lambda c: c.chapter_order)
        edges    = []

        for i, later in enumerate(sorted_c):
            e_later = MRLEmbedding(
                full_embedding=later.embedding, dims=self.encoder.dims
            )
            for earlier in sorted_c[:i]:
                if earlier.subject != later.subject:
                    continue
                gap = later.chapter_order - earlier.chapter_order
                if use_ordering and (gap <= 0 or gap > self.max_gap * 100):
                    continue

                conf   = 0.0
                method = []

                if use_mrl and earlier.embedding is not None:
                    e_early = MRLEmbedding(
                        full_embedding=earlier.embedding, dims=self.encoder.dims
                    )
                    s64  = self.encoder.compute_similarity(e_early, e_later, 64)
                    s768 = self.encoder.compute_similarity(e_early, e_later, 768)
                    agr  = self.encoder.multiscale_agreement_score(e_early, e_later)
                    if s64 >= self.coarse_thresh and s768 >= self.sim_thresh:
                        conf   = max(conf, 0.35*s64 + 0.35*s768 + 0.30*agr)
                        method.append("mrl")

                if use_linguistic:
                    lc = self._ling_conf(earlier.name, later.description)
                    if lc > 0:
                        conf   = max(conf, lc)
                        method.append("ling")

                if use_ordering and method:
                    conf = min(1.0, conf * 1.10)

                if conf >= self.min_conf:
                    edges.append(PrereqEdge(
                        earlier.concept_id, later.concept_id,
                        conf, "auto_" + "+".join(method), None,
                    ))
        return edges

    def _ling_conf(self, earlier_name: str, later_desc: str) -> float:
        m = PREREQ_LING_RE.search(later_desc)
        if m:
            matched = " ".join(g for g in m.groups() if g).lower()
            if earlier_name.lower() in matched:
                return 0.85
        if earlier_name.lower() in later_desc.lower():
            return 0.60
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────
class PrereqEvaluator:
    """
    Computes precision / recall / F1 of automated extraction vs human labels.

    Returns a results dict ready to paste into LaTeX tables.
    """

    def evaluate(self,
                 auto_edges:    List[PrereqEdge],
                 dataset:       AnnotationDataset,
                 threshold:     float = 0.60) -> Dict:

        pos_set: Set[Tuple[str,str]] = {
            (e.source_id, e.target_id) for e in dataset.positive_edges
        }
        neg_set: Set[Tuple[str,str]] = set(dataset.negative_pairs)
        all_pairs = pos_set | neg_set
        if not all_pairs:
            return {"error": "no evaluation pairs"}

        gt_ids = set(dataset.concept_map.values())
        auto_filtered = [
            e for e in auto_edges
            if e.source_id in gt_ids and e.target_id in gt_ids
        ]
        scores = {(e.source_id, e.target_id): e.confidence for e in auto_filtered}

        y_true, y_pred, y_score = [], [], []
        for pair in all_pairs:
            y_true.append(1 if pair in pos_set else 0)
            sc = scores.get(pair, 0.0)
            y_score.append(sc)
            y_pred.append(1 if sc >= threshold else 0)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        # Precision@K
        ranked = sorted(all_pairs,
                        key=lambda p: scores.get(p, 0.0), reverse=True)
        patk = {}
        for k in [10, 20, 50]:
            top = ranked[:k]
            patk[f"P@{k}"] = sum(1 for p in top if p in pos_set) / max(len(top),1)

        found = {p for p in auto_filtered
                 if (p.source_id, p.target_id) in pos_set
                 and p.confidence >= threshold}
        coverage = len(found) / max(len(pos_set), 1)

        return dict(
            precision=round(prec, 4), recall=round(rec, 4),
            f1=round(f1, 4), coverage=round(coverage, 4),
            n_gt_pos=len(pos_set), n_gt_neg=len(neg_set),
            n_pred_pos=sum(y_pred), n_auto=len(auto_filtered),
            **{k: round(v, 4) for k, v in patk.items()},
        )

    def ablation(self,
                 concepts: List[ConceptNode],
                 dataset:  AnnotationDataset,
                 encoder:  MRLEncoder,
                 config:   Dict) -> Dict:
        """
        Run the four-row ablation table from the paper.

        Row 1: ordering only
        Row 2: ordering + MRL
        Row 3: ordering + linguistic
        Row 4: ordering + MRL + linguistic  (full system)
        """
        ext = AutomatedExtractor(encoder, config)
        results = {}
        for name, mrl, ling in [
            ("ordering_only",       False, False),
            ("mrl_only",            True,  False),
            ("linguistic_only",     False, True),
            ("combined_full",       True,  True),
        ]:
            edges = ext.extract(concepts,
                                use_mrl=mrl, use_linguistic=ling,
                                use_ordering=True)
            r = self.evaluate(edges, dataset)
            r["n_edges"] = len(edges)
            results[name] = r

        self._print(results)
        return results

    def _print(self, results: Dict):
        COL = 18
        METHODS = ["ordering_only","mrl_only","linguistic_only","combined_full"]
        HDRS    = ["Method","Precision","Recall","F1","P@10","Coverage","#Edges"]
        print("\n" + "="*(COL*len(HDRS)))
        print("Prerequisite Extraction Ablation (vs Human Annotations)")
        print("="*(COL*len(HDRS)))
        print("".join(h.ljust(COL) for h in HDRS))
        print("-"*(COL*len(HDRS)))
        for m in METHODS:
            r = results.get(m, {})
            row = [m,
                   f"{r.get('precision',0):.4f}",
                   f"{r.get('recall',0):.4f}",
                   f"{r.get('f1',0):.4f}",
                   f"{r.get('P@10',0):.4f}",
                   f"{r.get('coverage',0):.4f}",
                   str(r.get('n_edges',0))]
            print("".join(v.ljust(COL) for v in row))
        print("="*(COL*len(HDRS)))


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────
class PrereqGraphPipeline:

    def __init__(self, encoder: MRLEncoder, config: Dict):
        self.encoder   = encoder
        self.config    = config
        self.evaluator = PrereqEvaluator()

    def run(self,
            corpus_concepts:  List[ConceptNode],
            lecturebank_dir:  Optional[str] = None,
            mooccube_dir:     Optional[str] = None,
            output_path:      Optional[str] = None) -> Dict:

        # 1. Load annotation datasets
        lb = LectureBankLoader(lecturebank_dir or "data/raw/lecturebank").load()
        mc_loader = MOOCCubeLoader(mooccube_dir or "data/raw/mooccube")
        mc_concepts = mc_loader.load_concepts()
        mc_edges    = mc_loader.load_edges()

        # 2. Ablation study on LectureBank ground truth
        logger.info("Running ablation on LectureBank...")
        ablation = self.evaluator.ablation(
            corpus_concepts, lb, self.encoder,
            self.config.get("knowledge_graph", {})
        )

        # 3. Automated extraction on full corpus
        extractor   = AutomatedExtractor(
            self.encoder, self.config.get("knowledge_graph", {})
        )
        auto_edges  = extractor.extract(corpus_concepts)
        lb_eval     = self.evaluator.evaluate(auto_edges, lb)

        # 4. Merge: human-annotated first, then high-confidence automated
        human_pairs = {
            (e.source_id, e.target_id)
            for e in lb.positive_edges + mc_edges
        }
        merged = list(lb.positive_edges) + list(mc_edges) + [
            e for e in auto_edges
            if e.confidence >= 0.80
            and (e.source_id, e.target_id) not in human_pairs
        ]

        logger.info(
            f"Final graph: {len(merged)} edges  "
            f"({len(human_pairs)} human + "
            f"{len(merged)-len(human_pairs)} high-conf automated)"
        )

        result = dict(
            verified_edges = [e.to_dict() for e in merged],
            evaluation     = {"lecturebank": lb_eval},
            ablation       = ablation,
            stats          = dict(
                total_edges       = len(merged),
                human_verified    = len(human_pairs),
                auto_added        = len(merged) - len(human_pairs),
                n_corpus_concepts = len(corpus_concepts),
                n_mc_concepts     = len(mc_concepts),
            ),
        )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved to {output_path}")

        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lecturebank", default=None)
    ap.add_argument("--mooccube",   default=None)
    ap.add_argument("--corpus",     default=None,
                    help="Path to processed concepts.json from OpenStax pipeline")
    ap.add_argument("--output",     default="data/processed/prereq_graph.json")
    ap.add_argument("--simulate",   action="store_true",
                    help="Run with simulated data (no downloads needed)")
    args = ap.parse_args()

    encoder = MRLEncoder({
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "matryoshka_dims": [64,128,256,512,768],
        "full_dim": 768, "batch_size": 32, "device": "cpu",
    })

    corpus_concepts = []
    if args.corpus:
        with open(args.corpus) as f:
            corpus_concepts = [ConceptNode.from_dict(d) for d in json.load(f)]

    pipeline = PrereqGraphPipeline(encoder, {"knowledge_graph": {}})
    result   = pipeline.run(
        corpus_concepts  = corpus_concepts,
        lecturebank_dir  = args.lecturebank,
        mooccube_dir     = args.mooccube,
        output_path      = args.output,
    )
    print(f"\nStats: {result['stats']}")