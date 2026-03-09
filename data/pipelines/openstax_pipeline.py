"""
OpenStax Real Data Pipeline
============================
Downloads and processes actual OpenStax textbooks (CC-BY licensed, free).

Usage:
    # First 5 chapters for quick test
    python data/pipelines/openstax_pipeline.py --book physics_v1 --max-chapters 5

    # Full book
    python data/pipelines/openstax_pipeline.py --book physics_v1

    # All paper books
    python data/pipelines/openstax_pipeline.py --book all

    # In Python
    from data.pipelines.openstax_pipeline import OpenStaxPipeline
    concepts, chunks = OpenStaxPipeline.load_processed("physics_v1")
"""

import os, re, json, time, hashlib, logging, argparse, requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_graph.graph_builder import ConceptNode, CorpusChunk
from data.processors.educational_processors import TextChunker, generate_id

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Book registry
# ─────────────────────────────────────────────────────────────────────────────
OPENSTAX_BOOKS = {
    "physics_v1": {
        "api_id":  "14fb4ad7-39a1-4eee-ab6e-3ef2482e3e22",
        "subject": "physics",
        "desc":    "University Physics Vol 1 — Mechanics, Waves, Thermo",
    },
    "physics_v2": {
        "api_id":  "7a0f9770-1c44-4acd-9920-1cd9a99f2a1e",
        "subject": "physics",
        "desc":    "University Physics Vol 2 — E&M, Optics",
    },
    "physics_hs": {
        "api_id":  "cce64fde-f448-43b8-ae88-27705cceb0da",
        "subject": "physics",
        "desc":    "High School Physics",
    },
    "calculus_v1": {
        "api_id":  "13ac107a-f15f-49d2-97e8-60ab2e3b519c",
        "subject": "calculus",
        "desc":    "Calculus Vol 1 — Limits, Derivatives, Integrals",
    },
    "chemistry_v2": {
        "api_id":  "85abf193-2bd2-4908-8563-90b8a7ac8df6",
        "subject": "chemistry",
        "desc":    "Chemistry 2e",
    },
}

ARCHIVE_BASE = "https://openstax.org/apps/archive/20230630.210259"
SKIP_TITLES  = {
    "preface","index","answer key","appendix","about openstax",
    "about this book","for the instructor","for the student",
}
OBJ_RE = re.compile(
    r"(?:explain|define|describe|calculate|derive|identify|apply|"
    r"distinguish|compare|evaluate|discuss)\s+"
    r"(?:the\s+|a\s+|an\s+)?(.{8,70}?)(?:\.|,|;|\s+and\s+|$)",
    re.IGNORECASE,
)
RECONTEX_RE = re.compile(
    r"recall\s+(from\s+)?(chapter|section|earlier)|"
    r"as\s+(we|you)\s+(saw|learned|discussed)|"
    r"building\s+on\s+(our|the|this)|"
    r"in\s+chapter\s+\d+|"
    r"we\s+(introduced|defined|derived)\s+the|"
    r"now\s+that\s+we\s+(understand|know)",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP downloader with disk cache
# ─────────────────────────────────────────────────────────────────────────────
class _Downloader:
    def __init__(self, cache_dir: str):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.sess = requests.Session()
        self.sess.headers["User-Agent"] = "PLEDGE-KARMA academic research"

    def _key(self, url: str) -> Path:
        return self.cache / f"{hashlib.md5(url.encode()).hexdigest()}.json"

    def get(self, url: str) -> Optional[Dict]:
        p = self._key(url)
        if p.exists():
            return json.loads(p.read_text())
        for i in range(3):
            try:
                r = self.sess.get(url, timeout=30)
                r.raise_for_status()
                p.write_text(r.text)
                time.sleep(0.4)
                return r.json()
            except Exception as e:
                logger.warning(f"Retry {i+1}/3 for {url}: {e}")
                time.sleep(2 ** i)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HTML parser
# ─────────────────────────────────────────────────────────────────────────────
def _parse_page(html: str) -> Dict:
    """Extract structured content from an OpenStax HTML page."""
    soup = BeautifulSoup(html, "lxml")
    out = dict(title="", objectives=[], glossary=[],
               paragraphs=[], examples=[], recontex=False)

    t = soup.find(attrs={"data-type": "document-title"}) or soup.find("h1")
    out["title"] = t.get_text(" ", strip=True) if t else ""

    ab = soup.find("div", {"data-type": "abstract"})
    if ab:
        out["objectives"] = [
            li.get_text(" ", strip=True)
            for li in ab.find_all("li") if len(li.get_text()) > 20
        ]

    gl = soup.find("section", {"data-type": "glossary"})
    if gl:
        for dt in gl.find_all("dt"):
            term = dt.get_text(" ", strip=True)
            dd   = dt.find_next_sibling("dd")
            defn = dd.get_text(" ", strip=True) if dd else ""
            if term and len(defn) > 10:
                out["glossary"].append({"term": term, "definition": defn})

    for ex in soup.find_all("div", {"data-type": "example"}):
        txt = ex.get_text(" ", strip=True)
        if len(txt) > 80:
            out["examples"].append(txt[:1500])

    SKIP_DTYPES = {"glossary", "footnotes", "abstract"}
    for p in soup.find_all("p"):
        parent_dtypes = {
            a.get("data-type", "") for a in p.parents if hasattr(a, "get")
        }
        if parent_dtypes & SKIP_DTYPES:
            continue
        txt = p.get_text(" ", strip=True)
        if len(txt) < 60:
            continue
        low = txt.lower()
        if re.match(r"^(figure|table|equation|check\s+your)\s*\d", low):
            continue
        out["paragraphs"].append(txt)
        if RECONTEX_RE.search(txt):
            out["recontex"] = True

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class OpenStaxPipeline:

    def __init__(self,
                 output_dir:   str = "data/processed",
                 cache_dir:    str = "data/raw/openstax_cache",
                 chunk_size:   int = 300,
                 max_chapters: Optional[int] = None):
        self.out_dir     = Path(output_dir)
        self.dl          = _Downloader(cache_dir)
        self.chunker     = TextChunker(chunk_size=chunk_size, overlap=50)
        self.max_chapters= max_chapters

    # ── public ────────────────────────────────────────────────────────────────
    def process_book(self, book_key: str
                     ) -> Tuple[List[ConceptNode], List[CorpusChunk]]:
        if book_key not in OPENSTAX_BOOKS:
            raise ValueError(f"Unknown book key '{book_key}'")
        meta    = OPENSTAX_BOOKS[book_key]
        api_id  = meta["api_id"]
        subject = meta["subject"]
        logger.info(f"Processing: {meta['desc']}")

        toc = self.dl.get(f"{ARCHIVE_BASE}/contents/{api_id}.json")
        if not toc:
            raise RuntimeError(f"Failed to fetch TOC for {book_key}")

        chapters = [
            item for item in toc.get("tree", {}).get("contents", [])
            if item.get("contents")
            and not any(s in item.get("title","").lower() for s in SKIP_TITLES)
        ]
        if self.max_chapters:
            chapters = chapters[:self.max_chapters]
        logger.info(f"  {len(chapters)} chapters")

        all_concepts: List[ConceptNode] = []
        all_chunks:   List[CorpusChunk] = []

        for ch_idx, chapter in enumerate(chapters):
            ch_num = ch_idx + 1
            depth  = self._depth(ch_num, len(chapters))
            logger.info(f"  Ch {ch_num}: {chapter.get('title','')[:60]}")

            for pg_idx, page in enumerate(chapter.get("contents", [])):
                pid = page.get("id", "")
                if not pid:
                    continue
                ch_order = ch_num * 100 + pg_idx
                raw = self.dl.get(f"{ARCHIVE_BASE}/contents/{api_id}:{pid}.json")
                if not raw or "content" not in raw:
                    continue

                parsed   = _parse_page(raw["content"])
                concepts = self._concepts(parsed, ch_order, depth, subject)
                chunks   = self._chunks(parsed, concepts, ch_order,
                                        depth, subject, pid, ch_num)
                all_concepts.extend(concepts)
                all_chunks.extend(chunks)

        all_concepts = self._dedup(all_concepts)
        logger.info(f"  Done: {len(all_concepts)} concepts, {len(all_chunks)} chunks")
        self._save(book_key, all_concepts, all_chunks)
        return all_concepts, all_chunks

    # ── concept extraction ────────────────────────────────────────────────────
    def _concepts(self, parsed, ch_order, depth, subject) -> List[ConceptNode]:
        out = []
        for g in parsed["glossary"]:
            term, defn = g["term"].strip(), g["definition"].strip()
            if len(term) > 2 and len(defn) > 14:
                out.append(ConceptNode(
                    concept_id    = generate_id(term.lower() + subject, "concept"),
                    name          = term,
                    description   = defn,
                    source_chunk_ids=[],
                    depth_level   = depth,
                    chapter_order = ch_order,
                    subject       = subject,
                    tags          = ["glossary", subject],
                ))
        for obj in parsed["objectives"]:
            m = OBJ_RE.search(obj)
            if m:
                name = m.group(1).strip().rstrip(".")
                if 6 < len(name) < 65:
                    out.append(ConceptNode(
                        concept_id    = generate_id(name.lower() + subject, "concept"),
                        name          = name,
                        description   = obj,
                        source_chunk_ids=[],
                        depth_level   = depth,
                        chapter_order = ch_order,
                        subject       = subject,
                        tags          = ["learning_objective", subject],
                    ))
        return out

    # ── chunk extraction ──────────────────────────────────────────────────────
    def _chunks(self, parsed, concepts, ch_order,
                depth, subject, pid, ch_num) -> List[CorpusChunk]:
        out   = []
        c_map = {c.name.lower(): c.concept_id for c in concepts}
        meta  = dict(chapter=ch_num, page_id=pid, title=parsed["title"],
                     source="openstax", subject=subject,
                     chapter_order=ch_order, depth_level=depth,
                     recontex=parsed["recontex"])

        full = " ".join(parsed["paragraphs"])
        if full.strip():
            for rc in self.chunker.chunk(full, meta):
                low  = rc["text"].lower()
                cids = [cid for nm, cid in c_map.items()
                        if len(nm) > 3 and nm in low]
                out.append(CorpusChunk(
                    chunk_id=generate_id(rc["text"][:100] + pid, "os"),
                    text=rc["text"], concept_ids=cids,
                    prerequisite_concept_ids=[],
                    depth_level=depth, chapter_order=ch_order,
                    subject=subject, source="openstax", metadata=meta,
                ))

        for i, ex in enumerate(parsed["examples"]):
            if len(ex) > 100:
                low  = ex.lower()
                cids = [cid for nm, cid in c_map.items()
                        if len(nm) > 3 and nm in low]
                out.append(CorpusChunk(
                    chunk_id=generate_id(ex[:80] + pid + str(i), "os_ex"),
                    text=ex, concept_ids=cids,
                    prerequisite_concept_ids=[],
                    depth_level=depth, chapter_order=ch_order,
                    subject=subject, source="openstax_example",
                    metadata={**meta, "is_example": True},
                ))
        return out

    # ── helpers ───────────────────────────────────────────────────────────────
    def _depth(self, ch_num: int, total: int) -> int:
        pos = ch_num / max(total, 1)
        return 0 if pos < 0.35 else (1 if pos < 0.70 else 2)

    def _dedup(self, concepts: List[ConceptNode]) -> List[ConceptNode]:
        seen: Dict[str, ConceptNode] = {}
        for c in concepts:
            k = c.name.lower().strip()
            if k not in seen or c.chapter_order < seen[k].chapter_order:
                seen[k] = c
        return list(seen.values())

    # ── persistence ───────────────────────────────────────────────────────────
    def _save(self, book_key, concepts, chunks):
        d = self.out_dir / book_key
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "concepts.json", "w") as f:
            json.dump([c.to_dict() for c in concepts], f, indent=2)
        with open(d / "chunks.json", "w") as f:
            json.dump([
                dict(chunk_id=c.chunk_id, text=c.text,
                     concept_ids=c.concept_ids,
                     prerequisite_concept_ids=c.prerequisite_concept_ids,
                     depth_level=c.depth_level, chapter_order=c.chapter_order,
                     subject=c.subject, source=c.source, metadata=c.metadata)
                for c in chunks
            ], f, indent=2)
        logger.info(f"  Saved to {d}")

    @classmethod
    def load_processed(cls, book_key: str,
                       data_dir: str = "data/processed"
                       ) -> Tuple[List[ConceptNode], List[CorpusChunk]]:
        d = Path(data_dir) / book_key
        with open(d / "concepts.json") as f:
            concepts = [ConceptNode.from_dict(x) for x in json.load(f)]
        with open(d / "chunks.json") as f:
            raw = json.load(f)
        chunks = [CorpusChunk(
            chunk_id=x["chunk_id"], text=x["text"],
            concept_ids=x["concept_ids"],
            prerequisite_concept_ids=x["prerequisite_concept_ids"],
            depth_level=x["depth_level"], chapter_order=x["chapter_order"],
            subject=x["subject"], source=x["source"],
            metadata=x.get("metadata", {}))
            for x in raw]
        logger.info(f"Loaded {book_key}: {len(concepts)} concepts, "
                    f"{len(chunks)} chunks")
        return concepts, chunks


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="physics_v1",
                    help=f"Book key or 'all'. Choices: {list(OPENSTAX_BOOKS)}")
    ap.add_argument("--output",       default="data/processed")
    ap.add_argument("--cache",        default="data/raw/openstax_cache")
    ap.add_argument("--max-chapters", type=int, default=None)
    ap.add_argument("--list",         action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in OPENSTAX_BOOKS.items():
            print(f"  {k:20s}  {v['desc']}")
        exit(0)

    pl = OpenStaxPipeline(args.output, args.cache,
                          max_chapters=args.max_chapters)
    books = list(OPENSTAX_BOOKS) if args.book == "all" else [args.book]
    for bk in books:
        c, ch = pl.process_book(bk)
        print(f"✓ {bk}: {len(c)} concepts, {len(ch)} chunks")