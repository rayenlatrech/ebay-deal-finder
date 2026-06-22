"""Test bootstrap.

Stubs the ``supabase`` package so the pure-logic modules (which import
``src.db`` at module load) can be imported without the real dependency or any
network/credentials. The classifier and statistical helpers under test never
touch the database.
"""

import sys
import types
from pathlib import Path

# Make the project root importable as `src.*`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "supabase" not in sys.modules:
    stub = types.ModuleType("supabase")
    stub.create_client = lambda *args, **kwargs: None
    stub.Client = object
    sys.modules["supabase"] = stub
