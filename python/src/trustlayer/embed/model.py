"""Model loader: EmbeddingGemma-2 text encoder, fp32, pinned revision.

Needs the ``embed`` extra.
"""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIM = 768
MAX_LENGTH = 512


def load_image_model(device: str = "cpu") -> SentenceTransformer:
    """Load the text+image encoder in float32 with eager attention.

    Args:
        device: ``cpu`` default — the image subset is 50 forms, and the
            text-run MPS stall showed long unattended MPS runs are risky.

    Audio tower nulled (Phase 11); vision tower kept (~440M params total).

    """
    return SentenceTransformer(
        MODEL_ID,
        revision=REVISION,
        device=device,
        config_kwargs={"audio_config": None},
        model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"},
    )


def load_text_model(device: str = "mps") -> SentenceTransformer:
    """Load the text-only encoder in float32 with eager attention.

    Args:
        device: ``mps`` (verified against CPU fp32 reference) or ``cpu``.

    Text-only selection happens at load (vision/audio configs nulled) —
    weights for unused towers are ignored, not downloaded separately.
    Eager attention avoids the SDPA-on-macOS NaN precedent; fp32 is the
    spec default (checkpoint is bf16; NEVER float16 per the model card).

    """
    return SentenceTransformer(
        MODEL_ID,
        revision=REVISION,
        device=device,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"},
    )


def embed_texts(
    model: SentenceTransformer, texts: list[str], batch_size: int = 32
) -> list[list[float]]:
    """Encode document texts with the Document prompt; normalized 768-d out.

    ``encode_document`` applies the ``title: none | text: `` prompt itself —
    callers pass raw ``title + body`` text, never a pre-formatted prefix
    (that would double-prefix).
    """
    vectors = model.encode_document(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        processing_kwargs={"text": {"max_length": MAX_LENGTH, "truncation": True}},
    )
    return [row.astype("float32").tolist() for row in vectors]
