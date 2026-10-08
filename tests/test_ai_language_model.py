"""The real language model on real questions. It needs the weights (1.75 GB).

Run it with: uv run pytest -m slow tests/test_ai_language_model.py
"""

import pytest

from ai.language import LanguageModel
from ai.settings import load_config

pytestmark = pytest.mark.slow

QUESTIONS = [
    ("what I watch movies on", ("tv", "television", "tablet", "laptop", "screen")),
    ("the thing I use to cut paper", ("scissors",)),
    ("something to sit on", ("chair", "sofa", "stool", "couch")),
    ("what I wear on my feet", ("shoe", "sandal", "slipper", "sock")),
    ("what keeps the rain off me", ("umbrella",)),
]


@pytest.fixture(scope="module")
def model() -> LanguageModel:
    return LanguageModel(load_config())


@pytest.mark.parametrize(("question", "right"), QUESTIONS)
def test_a_description_gives_a_right_name(
    model: LanguageModel, question: str, right: tuple[str, ...]
) -> None:
    names = model.rewrite_query(question)
    assert names and len(names) <= load_config().language.max_names
    assert any(any(word in name for word in right) for name in names), names
