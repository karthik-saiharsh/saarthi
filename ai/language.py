"""Language model (Qwen3.5-0.8B): it turns a description of an object into object names.

The user may describe an object instead of naming it: "that thing I check my mails on". Neither a sentence
encoder nor SigLIP connects "check my mails" with a laptop. A small language model knows it. The model gets the
description and writes two or three common names ("laptop, computer, tablet"). The open search then looks for these names.

The names only feed the search. They never reach the voice, and a name still needs a stored box that fits it.
The words stay on the device. The model is multimodal, so the same weights can later describe a frame.
"""

import logging
import re
import threading
import time

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from ai.errors import InferenceError, ModelLoadError
from ai.runtime import load_kwargs, resolve_device, short_model_name
from ai.settings import AiConfig

logger: logging.Logger = logging.getLogger(__name__)

# A name has letters, spaces, hyphens and apostrophes only, and at most this many words.
MAX_NAME_WORDS: int = 4
MAX_NAME_CHARS: int = 40
_NOT_NAME = re.compile(r"[^a-z' -]")


def clean_names(text: str, max_names: int, exclude: str = "") -> list[str]:
    """Turn the answer of the model into a list of names. Drop what is not a short everyday name.

    The model can write a number, a sentence or a long text. Only short names remain. The names are lower case,
    without a repeat, and without `exclude` (the words of the user).
    """
    assert isinstance(text, str), "text must be a string"
    assert max_names > 0, "max_names must be positive"
    names: list[str] = []
    for part in re.split(r"[,;\n]", text):
        name: str = " ".join(_NOT_NAME.sub(" ", part.lower()).split()).strip("-' ")
        if not name or len(name) > MAX_NAME_CHARS or len(name.split()) > MAX_NAME_WORDS:
            continue
        if name == exclude.lower().strip() or name in names:
            continue
        names.append(name)
    return names[:max_names]


class LanguageModel:
    """Rewrites a description into object names. Load it once."""

    def __init__(self, config: AiConfig) -> None:
        assert config is not None, "config is required"
        self._config: AiConfig = config
        self._device: str = resolve_device(config.models.device)
        self._lock: threading.Lock = threading.Lock()
        started: float = time.perf_counter()
        model_id: str = config.models.language_id
        revision: str = config.models.language_revision
        try:
            self._processor = AutoProcessor.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=config.models.local_files_only,
                trust_remote_code=False,
            )
            self._model = AutoModelForMultimodalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
                **load_kwargs(config.models, revision),
            )
        except (OSError, ValueError) as error:
            logger.error("Cannot load the language model %s: %s", model_id, error)
            raise ModelLoadError(f"Cannot load {model_id}") from error
        self._model.eval().to(self._device)
        self.name: str = short_model_name(model_id, revision)
        logger.info(
            "Language model %s ready on %s in %.1f s",
            self.name,
            self._device,
            time.perf_counter() - started,
        )

    def _messages(self, words: str) -> list[dict]:
        cfg = self._config.language
        messages: list[dict] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": cfg.rewrite_system}],
            }
        ]
        for question, answer in cfg.rewrite_examples:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": question}]}
            )
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            )
        messages.append({"role": "user", "content": [{"type": "text", "text": words}]})
        return messages

    def rewrite_query(self, words: str) -> list[str]:
        """Return up to `language.max_names` common names for the object that the words describe.

        The list can be empty. The words are cut to `language.max_input_chars` and never logged.
        """
        assert isinstance(words, str), "words must be a string"
        cfg = self._config.language
        text: str = " ".join(words.split())[: cfg.max_input_chars]
        if not text:
            return []
        try:
            prompt: str = self._processor.apply_chat_template(
                self._messages(text),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # The model answers at once. A long thought costs seconds.
            )
            inputs = self._processor(text=[prompt], return_tensors="pt").to(
                self._device
            )
            started: float = time.perf_counter()
            with self._lock, torch.inference_mode():
                output = self._model.generate(
                    **inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False
                )
            answer: str = self._processor.batch_decode(
                output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            logger.error("Rewrite failed: %s: %s", type(error).__name__, error)
            raise InferenceError("Rewrite failed") from error
        names: list[str] = clean_names(answer, cfg.max_names, exclude=text)
        logger.info(
            "Rewrite gave %d name(s) in %.1f s",
            len(names),
            time.perf_counter() - started,
        )
        return names
