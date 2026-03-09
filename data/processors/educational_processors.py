"""
Re-export module: data.processors.educational_processors

The canonical implementation lives at models/educational_processors.py.
This module re-exports all public names so that data pipelines
(openstax_pipeline.py, prereq_graph_pipeline.py) can import from
data.processors.educational_processors without modification.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `models` resolves
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.educational_processors import *          # noqa: F401,F403
from models.educational_processors import (          # noqa: F401
    generate_id,
    TextChunker,
    OpenStaxProcessor,
    CK12Processor,
    AssistmentsProcessor,
    LectureBankProcessor,
    MOOCCubeProcessor,
)
