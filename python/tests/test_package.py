"""Phase 1 placeholder: proves the package is installed and importable.

This is deliberately not `assert True`. A source-layout project can fail in ways
that still let source files exist on disk, so this checks the things that only
hold when the build configuration is actually correct: the package resolves, it
resolves to *this* copy under src/, and the installed distribution metadata
agrees with the version the module declares.
"""

from importlib.metadata import version

import trustlayer


def test_package_is_installed_and_versioned() -> None:
    assert trustlayer.__version__ == version("trustlayer")
    assert "src/trustlayer" in trustlayer.__file__.replace("\\", "/")
