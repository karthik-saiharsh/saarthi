"""Scene classifier (stage Q2): the room and the surface of an object, from fixed lists of words.

A SigLIP model scores a picture against one sentence for each word of a list. The scores are
turned into probabilities that add up to 1. The description can only use the words of the lists.
A wrong word from a closed list is a safer error than a text that a model invented
(see `benchmarks/scene/README.md`).

The model reads pictures and short fixed texts only. The sentences of the lists are encoded one
time, when the model loads. Then each picture costs one run of the image side of the model.
"""

import logging
import threading
import time

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, PreTrainedTokenizerFast

from ai.errors import InferenceError, ModelLoadError
from ai.runtime import load_kwargs, resolve_device, short_model_name
from ai.settings import AiConfig
from ai.types import Box

logger: logging.Logger = logging.getLogger(__name__)

# The text side of the model takes this number of tokens. This is the length that it learned with.
TEXT_TOKENS: int = 64


def context_box(box: Box, width: int, height: int, context: float) -> Box:
    """Return the box with space around it, cut at the edge of the picture.

    The space is `context` times the size of the box: on each side of it, and below it.
    Above the box, the space is half of this. So the picture shows what is under the object.
    """
    assert width > 0 and height > 0, "picture size must be positive"
    assert context >= 0.0, "context must not be negative"
    side: int = round(box.width * context)
    below: int = round(box.height * context)
    above: int = round(box.height * context / 2)
    return Box(
        max(0, box.x0 - side),
        max(0, box.y0 - above),
        min(width, box.x1 + side),
        min(height, box.y1 + below),
    )


class SceneClassifier:
    """Scores a picture against the lists of rooms and surfaces. Load it once."""

    def __init__(self, config: AiConfig) -> None:
        assert config is not None, "config is required"
        self._config: AiConfig = config
        self._device: str = resolve_device(config.models.device)
        self._lock: threading.Lock = threading.Lock()
        started: float = time.perf_counter()
        model_id: str = config.models.scene_id
        revision: str = config.models.scene_revision
        kwargs = load_kwargs(config.models, revision)
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=config.models.local_files_only,
                trust_remote_code=False,
            )
            # The repository has a fast tokenizer file. The default tokenizer needs the library
            # sentencepiece, and the project does not use it.
            self._tokenizer = PreTrainedTokenizerFast.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=config.models.local_files_only,
                trust_remote_code=False,
            )
            self._model = AutoModel.from_pretrained(model_id, **kwargs)
        except (OSError, ValueError) as error:
            logger.error("Cannot load the scene model %s: %s", model_id, error)
            raise ModelLoadError(f"Cannot load {model_id}") from error
        self._model.eval().to(self._device)
        self.name: str = short_model_name(model_id, revision)
        try:
            self._room_texts: torch.Tensor = self._encode_texts(config.scene.rooms)
            self._surface_texts: torch.Tensor = self._encode_texts(
                config.scene.surfaces
            )
        except (RuntimeError, ValueError) as error:
            logger.error("Cannot encode the word lists: %s", error)
            raise ModelLoadError("Cannot encode the word lists") from error
        logger.info(
            "Scene model %s ready on %s in %.1f s (%d rooms, %d surfaces)",
            self.name,
            self._device,
            time.perf_counter() - started,
            len(config.scene.rooms),
            len(config.scene.surfaces),
        )

    def _encode_texts(self, labels: tuple[str, ...]) -> torch.Tensor:
        """Return the unit-length text vectors of the sentences of a list. One row for each word."""
        sentences: list[str] = [
            self._config.scene.prompt.format(label=label) for label in labels
        ]
        tokens = self._tokenizer(
            sentences,
            padding="max_length",
            max_length=TEXT_TOKENS,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            vectors: torch.Tensor = self._model.get_text_features(
                input_ids=tokens["input_ids"]
            ).pooler_output
        return torch.nn.functional.normalize(vectors, dim=-1)

    def _probabilities(self, image: Image.Image, texts: torch.Tensor) -> list[float]:
        assert isinstance(image, Image.Image), "image must be a PIL image"
        try:
            pixels = self._processor(
                images=image.convert("RGB"), return_tensors="pt"
            ).to(self._device)
            with self._lock, torch.inference_mode():
                vector: torch.Tensor = self._model.get_image_features(
                    pixel_values=pixels["pixel_values"]
                ).pooler_output
                vector = torch.nn.functional.normalize(vector, dim=-1)
                # The same score as the full model: a scaled cosine and a bias.
                logits: torch.Tensor = (vector @ texts.T)[
                    0
                ] * self._model.logit_scale.exp() + self._model.logit_bias
                probabilities: list[float] = torch.softmax(logits, dim=0).tolist()
            return probabilities
        except (RuntimeError, ValueError) as error:
            logger.error("Scene scoring failed: %s: %s", type(error).__name__, error)
            raise InferenceError("Scene scoring failed") from error

    def room_scores(self, image: Image.Image) -> dict[str, float]:
        """Return the probability of each room of the list. The values add up to 1."""
        return dict(
            zip(self._config.scene.rooms, self._probabilities(image, self._room_texts))
        )

    def surface_scores(self, image: Image.Image) -> list[tuple[str, float]]:
        """Return the surfaces of the list with their probabilities, the best first."""
        scores = zip(
            self._config.scene.surfaces,
            self._probabilities(image, self._surface_texts),
        )
        return sorted(scores, key=lambda item: -item[1])
