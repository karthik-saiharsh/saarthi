"""Compare two ways to describe the place of an object in a frame (the planned stage Q2).

Usage: uv run python benchmarks/scene/compare.py DATASET_DIR OUT_DIR --method lists|vlm256|vlm500|detector

- `lists`: SigLIP chooses the best room and the best surface from fixed lists of words.
  The colour of the surface comes from the pixels around the object.
- `vlm256`, `vlm500`: SmolVLM writes one short sentence.
- `detector`: the detector of the application (Grounding DINO) looks for furniture. The room follows from the furniture that it finds.

For a photo with a labelled object (`benchmarks/calibration/labels.json`), the script describes the place
of the object. For other photos, it describes the scene. The result is `scene_<method>.json` in OUT_DIR.
This script is not part of the runtime.
"""

import argparse
import colorsys
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
from common import load_image

# The third method looks for furniture with the detector that the application already has.
FURNITURE: list[str] = [
    "a table",
    "a desk",
    "a bed",
    "a sofa",
    "a chair",
    "a shelf",
    "a window sill",
    "a railing",
    "a cupboard",
    "a refrigerator",
    "a sink",
    "a toilet",
    "a bathtub",
    "a stove",
    "a laptop",
    "a television",
]
SURFACE_FURNITURE: tuple[str, ...] = (
    "a table",
    "a desk",
    "a bed",
    "a sofa",
    "a chair",
    "a shelf",
    "a window sill",
    "a railing",
    "a cupboard",
)
ROOM_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("a bathroom", ("a toilet", "a bathtub")),
    ("a kitchen", ("a refrigerator", "a stove")),
    ("a bedroom", ("a bed",)),
    ("a living room", ("a sofa", "a television")),
    ("a study room", ("a desk", "a laptop")),
]
EVIDENCE_SCORE: float = 0.30

REVISIONS: dict[str, tuple[str, str]] = {
    "lists": (
        "google/siglip-base-patch16-224",
        "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    ),
    "detector": (
        "IDEA-Research/grounding-dino-tiny",
        "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    ),
    "vlm256": (
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        "7e3e67edbbed1bf9888184d9df282b700a323964",
    ),
    "vlm500": (
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        "a7da5b986cb59b408707209984f360a5f4ad7e47",
    ),
}
ROOMS: list[str] = [
    "a bedroom",
    "a living room",
    "a kitchen",
    "a bathroom",
    "a corridor",
    "a balcony",
    "a study room",
    "a dining room",
    "an outdoor area",
]
SURFACES: list[str] = [
    "a table",
    "a desk",
    "a shelf",
    "the floor",
    "a bed",
    "a chair",
    "a window sill",
    "a kitchen counter",
    "a sofa",
    "a railing",
    "a cupboard",
]
OBJECT_PROMPT: str = (
    "Where is the bottle in this picture? Say what surface it is on, the colour of the surface, "
    "and what kind of room it is. Answer in one short sentence."
)
# A second wording of the prompts. The first one gave answers of one word with the small model.
FULL_OBJECT_PROMPT: str = (
    "Describe this picture in one full sentence. Start with: The bottle is on"
)
FULL_SCENE_PROMPT: str = (
    "Describe this picture in one full sentence. Say what kind of room it is and what is in it."
)
SCENE_PROMPT: str = (
    "What kind of room or place is this? Describe the main surface in view. Answer in one short sentence."
)


def colour_name(rgb: tuple[float, float, float]) -> str:
    """Name a colour with a small fixed palette."""
    hue, saturation, value = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    degrees: float = hue * 360
    if value < 0.2:
        return "black"
    if saturation < 0.12:
        return "white" if value > 0.8 else "grey"
    if degrees < 15 or degrees >= 345:
        return "red"
    if degrees < 40:
        return "brown" if value < 0.55 else ("beige" if saturation < 0.4 else "orange")
    if degrees < 65:
        return "yellow"
    if degrees < 170:
        return "green"
    if degrees < 260:
        return "blue"
    return "purple" if degrees < 300 else "pink"


def surface_colour(
    image: Image.Image, box: tuple[int, int, int, int] | None
) -> str | None:
    """Name the colour of the surface under the object, from the pixels below and beside the box."""
    if box is None:
        return None
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    below = (
        max(0, x0 - width // 4),
        y1,
        min(image.width, x1 + width // 4),
        min(image.height, y1 + max(8, height // 4)),
    )
    region = (
        image.crop(below)
        if below[3] - below[1] > 4
        else image.crop((max(0, x0 - width // 2), y0, x0, y1))
    )
    pixels = np.asarray(region.convert("RGB")).reshape(-1, 3)
    return colour_name(tuple(float(v) for v in np.median(pixels, axis=0)))


def context_crop(
    image: Image.Image, box: tuple[int, int, int, int] | None
) -> Image.Image:
    """The box with the same space around it, so that the surface is in the picture."""
    if box is None:
        return image
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    return image.crop(
        (
            max(0, x0 - width),
            max(0, y0 - height // 2),
            min(image.width, x1 + width),
            min(image.height, y1 + height),
        )
    )


def run_lists(images: dict[str, Image.Image], boxes: dict) -> dict:
    from transformers import AutoImageProcessor, AutoModel, PreTrainedTokenizerFast

    repo, revision = REVISIONS["lists"]
    kwargs = {
        "revision": revision,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    image_processor = AutoImageProcessor.from_pretrained(repo, **kwargs)
    # The default tokenizer needs the library sentencepiece. The repository has a fast tokenizer file.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(repo, **kwargs)
    model = AutoModel.from_pretrained(repo, use_safetensors=True, **kwargs).eval()

    def probabilities_of(picture: Image.Image, labels: list[str]) -> list[float]:
        texts = [f"a photo of {label}" for label in labels]
        tokens = tokenizer(
            texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        pixels = image_processor(images=picture, return_tensors="pt")
        with torch.inference_mode():
            logits = model(
                input_ids=tokens["input_ids"], pixel_values=pixels["pixel_values"]
            ).logits_per_image[0]
        return [float(v) for v in torch.softmax(logits, dim=0)]

    def rank(picture: Image.Image, labels: list[str]) -> list[tuple[str, float]]:
        texts = [f"a photo of {label}" for label in labels]
        tokens = tokenizer(
            texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        pixels = image_processor(images=picture, return_tensors="pt")
        with torch.inference_mode():
            logits = model(
                input_ids=tokens["input_ids"], pixel_values=pixels["pixel_values"]
            ).logits_per_image[0]
        probabilities = torch.softmax(logits, dim=0)
        order = torch.argsort(probabilities, descending=True)[:3]
        return [(labels[i], round(float(probabilities[i]), 3)) for i in order]

    results = {}
    for name, image in images.items():
        box = boxes.get(name)
        started = time.perf_counter()
        room = rank(image, ROOMS)
        surface = rank(context_crop(image, box), SURFACES)
        results[name] = {
            "has_object": box is not None,
            "room": room,
            "room_all": dict(zip(ROOMS, probabilities_of(image, ROOMS))),
            "surface": surface,
            "colour": surface_colour(image, box),
            "seconds": round(time.perf_counter() - started, 2),
        }
        print(
            f"photo {name}: room {room[0]} surface {surface[0]} colour {results[name]['colour']} ({results[name]['seconds']} s)",
            flush=True,
        )
    return results


def run_detector(images: dict[str, Image.Image], boxes: dict) -> dict:
    from ai.detector import CandidateDetector
    from ai.postprocess import match_phrases
    from ai.settings import load_config

    detector = CandidateDetector(load_config())
    results = {}
    for name, image in images.items():
        started = time.perf_counter()
        raw = detector.detect(image, FURNITURE, (), False, False, False)
        evidence: dict[str, float] = {}
        found: list[tuple[str, float, tuple[int, int, int, int]]] = []
        for detection in raw:
            for phrase in match_phrases(detection.label, FURNITURE):
                if phrase in FURNITURE and detection.score >= EVIDENCE_SCORE:
                    box = (
                        detection.box.x0,
                        detection.box.y0,
                        detection.box.x1,
                        detection.box.y1,
                    )
                    found.append((phrase, round(detection.score, 2), box))
                    evidence[phrase] = max(
                        evidence.get(phrase, 0.0), round(detection.score, 2)
                    )
        room = next(
            (label for label, needs in ROOM_RULES if any(n in evidence for n in needs)),
            None,
        )
        surface = None
        if boxes.get(name) is not None:
            ox0, oy0, ox1, oy1 = boxes[name]
            centre = ((ox0 + ox1) / 2, oy1)
            holding = [
                (phrase, score, box)
                for phrase, score, box in found
                if phrase in SURFACE_FURNITURE
                and box[0] <= centre[0] <= box[2]
                and box[1] <= centre[1] <= box[3] + (oy1 - oy0) * 0.15
            ]
            if holding:
                surface = min(
                    holding, key=lambda f: (f[2][2] - f[2][0]) * (f[2][3] - f[2][1])
                )[0]
        results[name] = {
            "has_object": boxes.get(name) is not None,
            "evidence": evidence,
            "room": room,
            "surface": surface,
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(
            f"photo {name}: room {room} surface {surface} evidence {evidence} ({results[name]['seconds']} s)",
            flush=True,
        )
    return results


def run_vlm(
    method: str, images: dict[str, Image.Image], boxes: dict, style: str
) -> dict:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    repo, revision = REVISIONS[method]
    processor = AutoProcessor.from_pretrained(
        repo,
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
        do_image_splitting=False,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        repo,
        revision=revision,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        dtype=torch.float32,
    ).eval()
    results = {}
    for name, image in images.items():
        with_object = boxes.get(name) is not None
        if style == "full":
            prompt = FULL_OBJECT_PROMPT if with_object else FULL_SCENE_PROMPT
        else:
            prompt = OBJECT_PROMPT if with_object else SCENE_PROMPT
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        small = image.copy()
        small.thumbnail((512, 512))
        inputs = processor(text=text, images=[small], return_tensors="pt")
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        answer = processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0].strip()
        results[name] = {
            "has_object": boxes.get(name) is not None,
            "text": answer,
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(f"photo {name}: {answer!r} ({results[name]['seconds']} s)", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--method", required=True, choices=sorted(REVISIONS))
    parser.add_argument(
        "--style",
        default="short",
        choices=("short", "full"),
        help="the wording of the prompt (VLM only)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels = json.loads(
        (
            Path(__file__).resolve().parents[1] / "calibration" / "labels.json"
        ).read_text()
    )["boxes"]
    boxes = {name: tuple(found[0]) if found else None for name, found in labels.items()}
    images = {name: load_image(args.dataset, name) for name in sorted(labels, key=int)}
    started = time.perf_counter()
    results = (
        run_lists(images, boxes)
        if args.method == "lists"
        else (
            run_detector(images, boxes)
            if args.method == "detector"
            else run_vlm(args.method, images, boxes, args.style)
        )
    )
    suffix = f"_{args.style}" if args.method.startswith("vlm") else ""
    (args.out / f"scene_{args.method}{suffix}.json").write_text(
        json.dumps(results, indent=1)
    )
    print(f"done in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
