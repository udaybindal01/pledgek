"""
HuggingFace OpenStax Pipeline
==============================
Downloads and processes the HuggingFaceTB/openstax_paragraphs dataset.
This replaces the broken OpenStax archive API fetcher.

The dataset contains hierarchical JSON objects:
Book -> Chapters -> Sections -> Paragraphs

Usage:
    from data.pipelines.hf_openstax_pipeline import HFOpenStaxPipeline
    pipeline = HFOpenStaxPipeline()
    concepts, chunks = pipeline.process_dataset(max_books=1)
"""

import os
import re
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_graph.graph_builder import ConceptNode, CorpusChunk
from data.processors.educational_processors import generate_id

logger = logging.getLogger(__name__)

# Try to import datasets, if not available, we need it installed
try:
    from datasets import load_dataset
except ImportError:
    logger.error("The 'datasets' library is required to use the HFOpenStaxPipeline.")
    logger.error("Please run: pip install datasets")


class HFOpenStaxPipeline:
    """
    Downloads and parses the HuggingFaceTB/openstax_paragraphs dataset
    into PLEDGE-KARMA ConceptNodes and CorpusChunks.
    """

    def __init__(self, output_dir: str = "data/processed"):
        self.out_dir = Path(output_dir)
        self.concepts: List[ConceptNode] = []
        self.chunks: List[CorpusChunk] = []

    def _parse_hierarchy(self, node: dict, depth: int, chapter_order: int, book_title: str) -> None:
        """
        Recursively walk the HuggingFace parsed hierarchy.
        node can be a book, chapter, or section.
        """
        title_raw = node.get("title")
        if not title_raw:
            return
        title = str(title_raw).strip()
        if not title:
            return

        # Create a concept for this section/chapter
        concept_id = generate_id(f"{book_title}_{title}", "os_hf")
        
        # Don't create concepts for structural filler titles
        skip_titles = {"preface", "index", "introduction", "summary", 
                       "key terms", "chapter review", "glossary", "appendix"}
        if title.lower() in skip_titles:
            concept_id = None
            concept = None
        else:
            concept = ConceptNode(
                concept_id=concept_id,
                name=title,
                description=f"{book_title}: {title}",
                source_chunk_ids=[],
                depth_level=min(depth, 2),  # Map tree depth to cognitive depth
                chapter_order=chapter_order,
                subject=book_title.split()[0].lower(),
                tags=["openstax", "hf"]
            )
            self.concepts.append(concept)

        # Base case: Paragraphs at this node
        paragraph_raw = node.get("paragraph")
        if paragraph_raw and concept_id:
            # Clean paragraph text
            text = str(paragraph_raw).strip()
            if len(text) > 40 and not text.lower().startswith(("figure", "table", "equation")):
                chunk_id = generate_id(text, "chunk")
                chunk = CorpusChunk(
                    chunk_id=chunk_id,
                    text=text,
                    concept_ids=[concept_id],
                    prerequisite_concept_ids=[],
                    depth_level=concept.depth_level,
                    chapter_order=chapter_order,
                    subject=concept.subject,
                    source="openstax_hf_paragraphs",
                    metadata={"title": title}
                )
                self.chunks.append(chunk)
                concept.source_chunk_ids.append(chunk_id)

        # Recursive case 1: Sections
        sections = node.get("sections")
        if sections:
            for i, sec in enumerate(sections):
                self._parse_hierarchy(sec, depth + 1, chapter_order + (i * 10), book_title)

        # Recursive case 2: Chapters
        chapters = node.get("chapters")
        if chapters:
            for i, chap in enumerate(chapters):
                # Major chapter boundary, step chapter order significantly
                self._parse_hierarchy(chap, depth + 1, chapter_order + (i * 1000), book_title)

    def process_dataset(self, max_books: Optional[int] = 1, book_name_filter: str = "") -> Tuple[List[ConceptNode], List[CorpusChunk]]:
        """
        Load dataset from HuggingFace and process it into concepts and chunks.
        """
        logger.info("Loading HuggingFaceTB/openstax_paragraphs dataset...")
        try:
            ds = load_dataset("HuggingFaceTB/openstax_paragraphs", split="train")
        except Exception as e:
            logger.error(f"Failed to load dataset from HuggingFace: {e}")
            raise e

        books_processed = 0
        
        for item in ds:
            title = item.get("book_title", "Unknown Book")
            
            if book_name_filter and book_name_filter.lower() not in title.lower():
                continue
                
            chapters_raw = item.get("chapters", [])
            if not chapters_raw:
                continue
                
            chapters_data = chapters_raw if isinstance(chapters_raw, list) else json.loads(chapters_raw)

            logger.info(f"Processing book: {title} with {len(chapters_data)} chapters")
            
            # Start hierarchy processing at depth 0
            for i, chap in enumerate(chapters_data):
                self._parse_hierarchy(chap, depth=0, chapter_order=books_processed * 100000 + (i * 1000), book_title=title)
            
            books_processed += 1
            if max_books and books_processed >= max_books:
                break

        # Filter out empty concepts
        self.concepts = [c for c in self.concepts if c.source_chunk_ids]

        logger.info(f"HF OpenStax processing complete: {len(self.concepts)} concepts, {len(self.chunks)} chunks extracted.")
        return self.concepts, self.chunks

    def save_processed(self, output_name: str = "openstax_hf") -> None:
        """Save parsed concepts and chunks to standard JSON format."""
        if not self.concepts or not self.chunks:
            logger.warning("No concepts or chunks to save. Did you call process_dataset()?")
            return

        out = self.out_dir / output_name
        out.mkdir(parents=True, exist_ok=True)

        concepts_dict = []
        for c in self.concepts:
            concepts_dict.append({
                "concept_id": c.concept_id,
                "name": c.name,
                "description": c.description,
                "depth_level": c.depth_level,
                "chapter_order": c.chapter_order,
                "subject": c.subject,
                "tags": c.tags,
                "source_chunk_ids": c.source_chunk_ids,
            })

        chunks_dict = []
        for c in self.chunks:
            chunks_dict.append({
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

        with open(out / "concepts.json", "w") as f:
            json.dump(concepts_dict, f, indent=2)

        with open(out / "chunks.json", "w") as f:
            json.dump(chunks_dict, f, indent=2)

        logger.info(f"Saved to {out}: {len(concepts_dict)} concepts, {len(chunks_dict)} chunks")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = HFOpenStaxPipeline()
    pipeline.process_dataset(max_books=1)
    pipeline.save_processed("physics_v1")  # Save it under physics_v1 so data_loader sees it seamlessly
