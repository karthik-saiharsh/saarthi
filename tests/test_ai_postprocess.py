import dataclasses

import numpy as np
import pytest
from PIL import Image, ImageFilter

from ai.postprocess import (
    MAX_PROMPT_TOKENS,
    apply_crop_quality,
    match_phrase,
    match_phrases,
    plan_prompts,
    select_candidates,
)
from ai.settings import DetectionConfig, load_config
from ai.types import Box, Candidate, RawDetection

CONFIG: DetectionConfig = load_config().detection
VOCABULARY: list[str] = [
    "a bottle",
    "a pair of glasses",
    "a set of keys",
    "a remote control",
]
WIDTH, HEIGHT = 1000, 800


def raw(
    box: tuple[int, int, int, int], label: str, score: float = 0.5, source: str = "full"
) -> RawDetection:
    return RawDetection(Box(*box), score, label, source)  # type: ignore[arg-type]


def select(
    detections: list[RawDetection], config: DetectionConfig = CONFIG
) -> list[Candidate]:
    return select_candidates(detections, WIDTH, HEIGHT, VOCABULARY, config)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("a bottle", "a bottle"),
        ("bottle", "a bottle"),
        ("glasses", "a pair of glasses"),
        ("keys", "a set of keys"),
        ("remote", "a remote control"),
        ("control", "a remote control"),
        ("hand", "a hand"),
        ("screen", "a television screen"),
        ("frame", "a picture frame"),
        ("object", "an object"),
        ("spaceship", "spaceship"),
        ("", ""),
    ],
)
def test_match_phrase(label: str, expected: str) -> None:
    known: list[str] = [
        *VOCABULARY,
        "a hand",
        "a television screen",
        "a picture frame",
        "an object",
    ]
    assert match_phrase(label, known) == expected


def test_match_phrase_prefers_the_phrase_that_shares_more_words() -> None:
    known: list[str] = ["a bottle", "a water bottle"]
    assert match_phrase("water bottle", known) == "a water bottle"
    assert match_phrase("bottle", known) == "a bottle"  # Same words come first


def test_cap_per_phrase_keeps_the_best_boxes() -> None:
    config: DetectionConfig = dataclasses.replace(CONFIG, max_per_phrase=2)
    detections: list[RawDetection] = [
        raw((i * 100, 0, i * 100 + 50, 50), "a bottle", score=0.1 * (i + 1))
        for i in range(5)
    ]
    scores: list[float] = [c.det_score for c in select(detections, config)]
    assert scores == pytest.approx([0.5, 0.4])


def test_duplicate_boxes_merge_and_keep_the_union_of_labels() -> None:
    result: list[Candidate] = select(
        [
            raw((100, 100, 200, 300), "a bottle", 0.4, "full"),
            raw((102, 101, 201, 302), "keys", 0.7, "tile"),
        ]
    )
    assert len(result) == 1
    assert result[0].det_score == 0.7 and result[0].source == "tile"
    assert set(result[0].phrases) == {"a bottle", "a set of keys"}


def test_separate_boxes_stay_separate_and_are_sorted_by_score() -> None:
    result: list[Candidate] = select(
        [raw((0, 0, 50, 50), "a bottle", 0.3), raw((500, 500, 560, 580), "keys", 0.9)]
    )
    assert [c.det_score for c in result] == [0.9, 0.3]


def test_size_rules_and_clipping() -> None:
    result: list[Candidate] = select(
        [
            raw((0, 0, 900, 700), "a bottle"),  # Over 60 % of the frame
            raw((10, 10, 15, 100), "a bottle"),  # Narrower than min_box_side
            raw((-50, 400, 100, 500), "keys"),  # Cut by the left edge
            raw((2000, 2000, 2100, 2100), "keys"),  # Outside the image
        ]
    )
    assert [c.box for c in result] == [Box(0, 400, 100, 500)]


def test_object_inside_a_depiction_is_marked_not_good_and_a_partial_overlap_is_good() -> (
    None
):
    result: list[Candidate] = select(
        [
            raw((100, 100, 700, 500), "a television screen", 0.8),
            raw((200, 200, 260, 300), "a bottle", 0.6),  # Inside the screen
            raw((680, 200, 760, 300), "keys", 0.6),  # 25 % inside the screen
        ]
    )
    good: dict[str, bool] = {c.phrases[0]: c.quality_ok for c in result}
    assert good == {"a bottle": False, "a set of keys": True}
    assert all("a television screen" not in c.phrases for c in result)


def test_crop_quality_does_not_make_a_depicted_candidate_good_again() -> None:
    depicted: Candidate = Candidate(
        Box(100, 100, 300, 300), 0.5, "full", ("a bottle",), False, False
    )
    result: list[Candidate] = apply_crop_quality([depicted], sharp_image(), 24, 20.0)
    assert result[0].quality_ok is False


def test_hand_flag_and_hand_is_not_a_candidate() -> None:
    result: list[Candidate] = select(
        [
            raw((100, 100, 200, 200), "a hand", 0.9),
            raw((150, 120, 230, 180), "a bottle", 0.6),  # About 50 % inside the hand
            raw((400, 400, 450, 450), "keys", 0.6),
        ]
    )
    flags: dict[str, bool] = {c.phrases[0]: c.in_hand for c in result}
    assert flags == {"a bottle": True, "a set of keys": False}


def test_generic_boxes_keep_the_generic_phrase() -> None:
    result: list[Candidate] = select(
        [raw((300, 300, 380, 380), "object", 0.3, "generic")]
    )
    assert result[0].phrases == ("an object",) and result[0].source == "generic"


def test_empty_label_gives_a_candidate_without_phrases() -> None:
    result: list[Candidate] = select([raw((300, 300, 380, 380), "", 0.3)])
    assert result[0].phrases == ()


def test_no_detections_give_no_candidates() -> None:
    assert select([]) == []


def sharp_image() -> Image.Image:
    rng: np.random.Generator = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (800, 1000, 3), dtype=np.uint8))


def candidate(box: Box) -> Candidate:
    return Candidate(box, 0.5, "full", ("a bottle",), False, True)


def test_crop_quality_rules() -> None:
    image: Image.Image = sharp_image()
    blurred: Image.Image = image.copy()
    blurred.paste(
        image.crop((300, 300, 500, 500)).filter(ImageFilter.GaussianBlur(15)),
        (300, 300),
    )
    cases: list[tuple[Image.Image, Box, bool]] = [
        (image, Box(100, 100, 200, 200), True),
        (image, Box(100, 100, 110, 200), False),  # Too narrow
        (image, Box(0, 100, 100, 200), False),  # Touches the edge
        (blurred, Box(320, 320, 480, 480), False),  # Blurry crop
    ]
    for source, box, expected in cases:
        result: list[Candidate] = apply_crop_quality([candidate(box)], source, 24, 20.0)
        assert result[0].quality_ok is expected, box


def fake_tokens(phrase: str) -> int:
    return len(phrase.split())


def test_plan_prompts_respects_the_phrase_limit() -> None:
    objects: list[str] = [f"a thing{i}" for i in range(7)]
    extras: list[str] = ["a hand", "a mirror"]
    prompts: list[list[str]] = plan_prompts(objects, extras, 5, fake_tokens)
    assert all(len(p) <= 5 for p in prompts)
    assert all(p[-2:] == extras for p in prompts)
    assert [p for prompt in prompts for p in prompt if p not in extras] == objects


def test_plan_prompts_respects_the_token_limit() -> None:
    objects: list[str] = ["word " * 40 for _ in range(10)]  # 40 tokens each
    prompts: list[list[str]] = plan_prompts(objects, ["a hand"], 30, fake_tokens)
    assert len(prompts) > 1
    for prompt in prompts:
        assert 2 + sum(fake_tokens(p) + 1 for p in prompt) <= MAX_PROMPT_TOKENS


def test_plan_prompts_single_prompt_when_everything_fits() -> None:
    assert len(plan_prompts(["a bottle", "a cup"], ["a hand"], 30, fake_tokens)) == 1


def test_plan_prompts_refuses_extras_that_fill_the_prompt() -> None:
    with pytest.raises(ValueError):
        plan_prompts(["a cup"], ["word " * 300], 30, fake_tokens)


KNOWN: list[str] = [
    *VOCABULARY,
    "a hand",
    "a mirror",
    "a television screen",
    "an object",
]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("a mobile phone a mirror", ["a mobile phone", "a mirror"]),
        ("a bottle a hand a mirror", ["a bottle", "a hand", "a mirror"]),
        ("a a hand", ["a hand"]),
        ("a pair of glasses a set of keys", ["a pair of glasses", "a set of keys"]),
        ("bottle", ["a bottle"]),
        ("a bottle a bottle", ["a bottle"]),
        ("", []),
        ("a", []),
    ],
)
def test_match_phrases_splits_a_joined_label(label: str, expected: list[str]) -> None:
    vocabulary: list[str] = [*KNOWN, "a mobile phone"]
    assert match_phrases(label, vocabulary) == expected


def test_a_box_with_an_object_phrase_stays_an_object_even_with_a_screen_phrase() -> (
    None
):
    # The real detector joins weak extra phrases to the label of a good box.
    result: list[Candidate] = select(
        [raw((373, 278, 453, 389), "a bottle a hand a mirror", 0.44)]
    )
    assert [c.phrases for c in result] == [("a bottle",)]
    assert not result[0].in_hand


def test_a_box_without_an_object_phrase_is_a_hand_or_a_screen() -> None:
    result: list[Candidate] = select(
        [
            raw((100, 100, 200, 200), "a a hand", 0.5),
            raw((120, 120, 160, 160), "bottle", 0.6),
            raw((600, 100, 760, 300), "a mirror", 0.5),
            raw((620, 120, 700, 200), "keys", 0.6),
        ]
    )
    assert {c.phrases[0]: c.in_hand for c in result} == {
        "a bottle": True,
        "a set of keys": False,
    }
    # The keys lie inside the mirror: kept, but not good.
    assert {c.phrases[0]: c.quality_ok for c in result} == {
        "a bottle": True,
        "a set of keys": False,
    }


def test_a_box_that_covers_most_of_the_frame_is_dropped_for_every_kind() -> None:
    result: list[Candidate] = select(
        [
            raw((5, 21, 962, 722), "a hand a mirror", 0.18),  # Over 60 % of the frame
            raw((300, 300, 380, 380), "a bottle", 0.5),
        ]
    )
    assert [c.phrases for c in result] == [("a bottle",)]
    assert not result[0].in_hand
