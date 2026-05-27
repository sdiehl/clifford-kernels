import py_compile
import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# Examples that should execute cleanly on a plain CPU CI runner. Anything
# requiring CUDA, network, or optional dependencies stays out of this list and
# is only byte-compiled below.
RUNNABLE = ["simple.py", "torch_compile.py"]


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
def test_example_compiles(path):
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("name", RUNNABLE)
def test_example_runs(name):
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
