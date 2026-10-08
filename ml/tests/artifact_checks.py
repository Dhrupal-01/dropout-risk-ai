"""
Tests marked `artifacts` read committed (git-tracked) generated files: benchmark/simulation JSON and
rendered docs. A missing file means a broken checkout, so these tests fail with the command that
regenerates it instead of skipping. To run without the artifacts, deselect: pytest -m "not artifacts".
"""

import pytest


def require_artifact(present: bool, what: str, command: str) -> None:
    if not present:
        pytest.fail(
            f"{what} is missing. It is a tracked generated artifact: restore it with git or run `{command}`. "
            'To run the suite without artifacts, deselect them with -m "not artifacts".',
            pytrace=False,
        )
