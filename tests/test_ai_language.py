"""The language model that rewrites a description into object names. The model is a stub, so the tests are fast."""

import dataclasses
import threading
from types import SimpleNamespace

import pytest
import torch

from ai.errors import InferenceError
from ai.language import LanguageModel, clean_names
from ai.settings import load_config
from tests.ai_support import FakeLanguageModel, make_engine

CONFIG = load_config()


# ---- The cleaning of the answer ----


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("laptop, computer, tablet", ["laptop", "computer", "tablet"]),
        ("Laptop,  COMPUTER\n tablet", ["laptop", "computer", "tablet"]),
        ("laptop, laptop, computer", ["laptop", "computer"]),
        ("1. laptop; 2. tablet", ["laptop", "tablet"]),
        ("water bottle, can't opener", ["water bottle", "can't opener"]),
        ("a b c d e, phone", ["phone"]),  # A name has at most 4 words
        ("x" * 50 + ", phone", ["phone"]),  # A name is short
        ("", []),
        (",,, ;;", []),
    ],
)
def test_the_answer_becomes_short_clean_names(text: str, names: list[str]) -> None:
    assert clean_names(text, 3) == names


def test_only_the_most_names_are_kept_and_the_words_of_the_user_are_left_out() -> None:
    assert clean_names("a, b, c, d, e", 3) == ["a", "b", "c"]
    assert clean_names("laptop, computer", 3, exclude="Laptop") == ["computer"]


# ---- The model (a stub) ----


class StubProcessor:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.messages: list = []

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, enable_thinking
    ):
        assert tokenize is False and add_generation_prompt and enable_thinking is False
        self.messages = messages
        return "PROMPT"

    def __call__(self, text, return_tensors):
        assert text == ["PROMPT"] and return_tensors == "pt"
        tokens = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

        class Batch(dict):
            def to(self, device):
                return self

        return Batch(tokens)

    def batch_decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return [self.answer]


def make_model(
    answer: str = "laptop, computer, tablet",
) -> tuple[LanguageModel, StubProcessor]:
    model = object.__new__(LanguageModel)  # No model is loaded
    processor = StubProcessor(answer)
    model._config = CONFIG
    model._device = "cpu"
    model._lock = threading.Lock()
    model._processor = processor
    model._model = SimpleNamespace(
        generate=lambda **kwargs: torch.zeros(1, 9, dtype=torch.long)
    )
    return model, processor


def test_a_description_gives_names_and_the_prompt_has_the_examples() -> None:
    model, processor = make_model()
    assert model.rewrite_query("that thing I check my mails on") == [
        "laptop",
        "computer",
        "tablet",
    ]
    roles = [m["role"] for m in processor.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert (
        processor.messages[-1]["content"][0]["text"] == "that thing I check my mails on"
    )
    assert "3 most likely" in processor.messages[0]["content"][0]["text"]


def test_the_words_are_cleaned_and_cut_before_the_model_gets_them() -> None:
    model, processor = make_model()
    model.rewrite_query("  many    spaces\n here " + "x" * 1000)
    sent = processor.messages[-1]["content"][0]["text"]
    assert (
        sent.startswith("many spaces here ")
        and len(sent) == CONFIG.language.max_input_chars
    )


def test_empty_words_give_no_names_and_no_model_call() -> None:
    model, processor = make_model()
    assert model.rewrite_query("   ") == []
    assert processor.messages == []


def test_a_model_error_becomes_an_inference_error_without_the_words() -> None:
    model, _ = make_model()

    def fail(**kwargs: object) -> None:
        raise RuntimeError("secret detail")

    model._model = SimpleNamespace(generate=fail)
    with pytest.raises(InferenceError) as caught:
        model.rewrite_query("my private words")
    assert "secret" not in str(caught.value) and "private" not in str(caught.value)


def test_an_answer_with_no_letters_gives_an_empty_list() -> None:
    model, _ = make_model("12345 !!!")
    assert model.rewrite_query("something") == []


# ---- The engine ----


def test_the_engine_gives_the_names_of_the_model() -> None:
    fake = FakeLanguageModel({"that thing": ["laptop"]})
    engine = make_engine(language_model=fake)
    assert engine.language_enabled and engine.rewrite_query("that thing") == ["laptop"]
    assert fake.calls == ["that thing"]


def test_a_switched_off_language_model_is_reported() -> None:
    engine = make_engine()
    engine._language = None
    assert not engine.language_enabled
    with pytest.raises(AssertionError, match="switched off"):
        engine.rewrite_query("words")


# ---- The description of a frame ----


class StubDescribeProcessor(StubProcessor):
    def __init__(self, answer: str) -> None:
        super().__init__(answer)
        self.images: list = []

    def __call__(self, text, return_tensors, images=None):
        self.images = images
        return super().__call__(text, return_tensors)


def make_describer(
    answer: str = "The bottle is on a table.",
) -> tuple[LanguageModel, StubDescribeProcessor]:
    model, _ = make_model()
    processor = StubDescribeProcessor(answer)
    model._processor = processor
    return model, processor


def test_a_description_gets_the_picture_and_the_facts() -> None:
    from PIL import Image

    model, processor = make_describer("  The bottle   is on a table.\n")
    picture = Image.new("RGB", (1600, 800))
    sentence = model.describe_scene(picture, ["Object: bottle", "It is on: a table"])
    assert sentence == "The bottle is on a table."
    user = processor.messages[-1]["content"]
    assert (
        user[0]["type"] == "image"
        and max(user[0]["image"].size) == CONFIG.language.describe_image_side
    )
    assert (
        "- Object: bottle" in user[1]["text"]
        and "- It is on: a table" in user[1]["text"]
    )
    assert processor.images and picture.size == (
        1600,
        800,
    )  # The picture of the caller is not changed


def test_a_model_error_in_a_description_becomes_an_inference_error() -> None:
    from PIL import Image

    model, _ = make_describer()

    def fail(**kwargs: object) -> None:
        raise RuntimeError("secret detail")

    model._model = SimpleNamespace(generate=fail)
    with pytest.raises(InferenceError) as caught:
        model.describe_scene(Image.new("RGB", (10, 10)), ["Object: x"])
    assert "secret" not in str(caught.value)


def test_the_engine_describes_and_names_the_model_with_a_fingerprint() -> None:
    from PIL import Image

    fake = FakeLanguageModel()
    fake.description = "A sentence."
    engine = make_engine(language_model=fake)
    assert (
        engine.describe_scene(Image.new("RGB", (20, 10)), ["Object: x"])
        == "A sentence."
    )
    assert fake.described == [((20, 10), ["Object: x"])]
    assert engine.description_model.startswith(fake.name + "/")
    changed = dataclasses.replace(
        CONFIG,
        language=dataclasses.replace(
            CONFIG.language, describe_system="Another instruction."
        ),
    )
    other = make_engine(language_model=fake)
    other._config = changed
    assert other.description_model != engine.description_model


def test_a_switched_off_language_model_cannot_describe() -> None:
    from PIL import Image

    engine = make_engine()
    engine._language = None
    with pytest.raises(AssertionError, match="switched off"):
        engine.describe_scene(Image.new("RGB", (10, 10)), ["Object: x"])
    with pytest.raises(AssertionError, match="switched off"):
        _ = engine.description_model
