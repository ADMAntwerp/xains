"""HuggingFacePerplexityProvider — local autoregressive perplexity via transformers + torch.

Computes ``exp(cross_entropy_loss)`` over the input text by running an
autoregressive language model with ``labels=input_ids``. The model and
tokenizer are eager-loaded in ``__init__`` (fail fast on missing weights);
``compute()`` then runs forward passes only.

Optional dependency: ``pip install "xains[perplexity-hf]"``
(transformers + torch).
"""

import math
import warnings
from typing import Any

_MISSING_HF_MESSAGE = (
    "The 'transformers' and 'torch' packages are required for "
    "HuggingFacePerplexityProvider. "
    'Install with: pip install "xains[perplexity-hf]"'
)


class HuggingFacePerplexityProvider:
    """Local-model perplexity via Hugging Face transformers + torch.

    The default ``model_name="gpt2"`` triggers a ~500 MB cached download on
    first use, which is fine for "let me play around" but inappropriate for
    paper replication or production scoring. Callers should set
    ``model_name`` explicitly:

    * Paper replication (Cedro & Martens 2026):
      ``model_name="meta-llama/Llama-3.1-8B"`` (requires accepting the
      Hugging Face gating for that model).
    * Tiny test deployments / CI:
      ``model_name="sshleifer/tiny-gpt2"`` (~10 MB).

    All weights are eager-loaded in ``__init__`` so deployment failures
    surface immediately rather than on the first ``compute()`` call. Long
    texts exceeding ``max_length`` are truncated from the right and emit a
    single ``UserWarning`` per provider instance.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str | None = None,
        max_length: int | None = None,
    ) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ImportError(_MISSING_HF_MESSAGE) from exc

        self._torch: Any = torch
        self._device: str = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._tokenizer: Any = transformers.AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        self._model: Any = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self._model.to(self._device)  # type: ignore[arg-type]
        self._model.eval()  # type: ignore[no-untyped-call]

        if max_length is not None:
            self._max_length = max_length
        else:
            self._max_length = int(getattr(self._model.config, "n_positions", 1024))

        self._truncation_warned: bool = False

    def compute(self, text: str) -> float | None:
        """Return the perplexity of ``text``, or ``None`` for degenerate inputs."""
        if not text.strip():
            return None

        encoded = self._tokenizer(text, return_tensors="pt", truncation=False)
        original_length = int(encoded["input_ids"].shape[1])
        input_ids = encoded["input_ids"]

        if original_length > self._max_length:
            input_ids = input_ids[:, : self._max_length]
            if not self._truncation_warned:
                warnings.warn(
                    f"HuggingFacePerplexityProvider: input length "
                    f"({original_length} tokens) exceeds max_length "
                    f"({self._max_length}); truncating from the right. "
                    f"Further truncations on this provider will not be warned.",
                    UserWarning,
                    stacklevel=2,
                )
                self._truncation_warned = True

        if input_ids.shape[1] < 2:
            return None

        input_ids = input_ids.to(self._device)

        with self._torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)

        loss = float(outputs.loss.item())

        try:
            ppl = math.exp(loss)
        except OverflowError:
            return None

        if math.isnan(ppl) or math.isinf(ppl):
            return None

        return ppl
