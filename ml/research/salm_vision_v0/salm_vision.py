"""Vision-lane plumbing for LFM2.5-Audio-1.5B (SigLIP2 tower transplant).

`LFM2AudioModel._prefill` scatters per-modality continuous vectors into the
token stream via modality masks. This module extends that mechanism with an
IMAGE flag: projected patch embeddings (tower + projector transplanted
verbatim from LiquidAI/LFM2.5-VL-1.6B) enter exactly where `<image>` (id 396)
sits in the token stream, mirroring how conformer features ride AUDIO_IN.

Row convention is unchanged: `text` holds ONLY text positions; IMAGE-flagged
slots are excluded from `text` and supplied as continuous vectors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING

import torch
from datasets import load_from_disk
from einops import rearrange
from liquid_audio.utils import LFMModality, mel2emb_len
from transformers import Lfm2VlForConditionalGeneration

if TYPE_CHECKING:
    from liquid_audio import LFM2AudioModel

IMAGE = 4  # extends IntEnum LFMModality (TEXT=1, AUDIO_IN=2, AUDIO_OUT=3)
RANGE_LO, RANGE_HI = 396, 500
VL_REPO = "LiquidAI/LFM2.5-VL-1.6B"


# ---------------------------------------------------------------- model patch

def _vision_features(
    self,
    pixel_values: torch.Tensor,
    spatial_shapes: torch.Tensor,
    pixel_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mirror of Lfm2VlModel.get_image_features: tower (no grad; frozen) ->
    unpad -> feature grid -> projector (2x2 pixel unshuffle + MLP) -> flat."""
    dtype = self.lfm.embed_tokens.weight.dtype
    with torch.no_grad():
        out = self.vision_tower(
            pixel_values=pixel_values.to(dtype),
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            return_dict=True,
        )
    hs = out.last_hidden_state
    feat_lens = pixel_attention_mask.sum(dim=1)
    out_dim = self.multi_modal_projector.linear_2.out_features
    feats: list[torch.Tensor] = []
    for i in range(hs.shape[0]):
        f = hs[i][: feat_lens[i], :].unsqueeze(0)
        h, w = (int(v) for v in spatial_shapes[i].tolist())
        f = f.reshape(1, h, w, -1)
        feats.append(self.multi_modal_projector(f).reshape(-1, out_dim))
    return torch.cat(feats, dim=0).to(dtype)


def _vision_prefill(
    self,
    *,
    text: torch.Tensor,
    audio_in: torch.Tensor,
    audio_in_lens: torch.Tensor,
    audio_out: torch.Tensor,
    modality_flag: torch.Tensor,
    image_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Original LFM2AudioModel._prefill with an IMAGE branch; audio asserts
    apply only when audio is present (vision/corrector rows carry none)."""
    assert len(text.shape) == 2 and len(modality_flag.shape) == 2 and text.shape[0] == 1
    assert (modality_flag == LFMModality.TEXT).sum() == text.shape[1]
    assert (modality_flag == LFMModality.AUDIO_OUT).sum() == audio_out.shape[1]

    text_emb = self.lfm.embed_tokens(text[0])
    text_mask = modality_flag == LFMModality.TEXT

    audio_in_mask = modality_flag == LFMModality.AUDIO_IN
    if audio_in_mask.any():
        assert audio_in.shape[0] == 128
        assert audio_in.shape[1] == audio_in_lens.sum()
        assert audio_in_mask.sum() == mel2emb_len(audio_in_lens).sum()
        audio_in_list = audio_in.mT.split(audio_in_lens.tolist())
        padded = torch.nn.utils.rnn.pad_sequence(audio_in_list, batch_first=True)
        audio_enc, audio_in_len = self.conformer(padded.mT.to(text_emb.dtype), audio_in_lens)
        len_mask = torch.arange(audio_enc.shape[-1], device=audio_enc.device).unsqueeze(0) < audio_in_len.unsqueeze(1)
        audio_in_emb = self.audio_adapter(audio_enc.mT[len_mask])
        assert audio_in_emb.shape[0] == audio_in_mask.sum()
    else:
        audio_in_emb = text_emb.new_empty((0, text_emb.shape[-1]))

    audio_out_mask = modality_flag == LFMModality.AUDIO_OUT

    image_mask = modality_flag == IMAGE
    if image_mask.any():
        assert image_embeds is not None, "IMAGE flags without features"
        assert image_embeds.shape[0] == image_mask.sum(), (
            f"{image_embeds.shape[0]} features vs {int(image_mask.sum())} IMAGE slots")

    B, L, D = *modality_flag.shape, self.lfm.config.hidden_size
    in_emb = text_emb.new_empty((B, L, D))
    in_emb[text_mask] = text_emb
    if audio_in_mask.any():
        in_emb[audio_in_mask] = audio_in_emb
    if image_mask.any():
        in_emb[image_mask] = image_embeds
    if audio_out_mask.any():
        offset = audio_out[: self.codebooks] + self.codebook_offsets.unsqueeze(1)
        audio_out_emb = self.audio_embedding(offset).sum(0, dtype=text_emb.dtype)
        assert audio_out_emb.shape[0] == audio_out_mask.sum()
        in_emb[audio_out_mask] = audio_out_emb

    return in_emb


def _vision_logits(self, batch):
    """Faithful copy of LFM2AudioModel.logits (depthformer path included, so
    the stock forward() works unchanged) with one addition: pixel tensors on
    the batch are run through the transplanted tower/projector and injected
    at IMAGE slots. Audio/corrector batches have no pixel_values and take the
    identical path as the unpatched model."""
    image_embeds = None
    if getattr(batch, "pixel_values", None) is not None:
        image_embeds = self._vision_features(
            batch.pixel_values, batch.spatial_shapes, batch.pixel_attention_mask)

    in_emb = self._prefill(
        text=batch.text,
        audio_in=batch.audio_in,
        audio_in_lens=batch.audio_in_lens,
        audio_out=batch.audio_out,
        modality_flag=batch.modality_flag,
        image_embeds=image_embeds,
    )

    out_emb = self.lfm(inputs_embeds=in_emb, use_cache=False).last_hidden_state
    out_emb_shifted = out_emb[:, :-1]

    text_mask = batch.modality_flag == LFMModality.TEXT
    audio_out_mask = batch.modality_flag == LFMModality.AUDIO_OUT

    supervised_text_mask = torch.logical_and(text_mask, batch.supervision_mask)[:, 1:]
    text_out_emb = out_emb_shifted[supervised_text_mask]
    text_logits = torch.nn.functional.linear(text_out_emb, self.lfm.embed_tokens.weight)

    shifted_text_mask = torch.logical_and(text_mask, batch.supervision_mask).clone()
    shifted_text_mask[:, 0] = False
    shifted_text_tokens = batch.text[0, shifted_text_mask[text_mask]]

    supervised_audio_mask = torch.logical_and(audio_out_mask, batch.supervision_mask)[:, 1:]
    audio_output_embeddings = out_emb_shifted[supervised_audio_mask]

    shifted_audio_mask = torch.logical_and(audio_out_mask, batch.supervision_mask).clone()
    shifted_audio_mask[:, 0] = False
    shifted_audio_tokens = batch.audio_out[: self.codebooks, shifted_audio_mask[audio_out_mask]]

    depthformer_in = rearrange(
        self.depth_linear(audio_output_embeddings),
        "L (C D) -> L C D",
        C=self.codebooks,
        D=self.depthformer_dim,
    )

    depthformer_tokens = torch.stack(
        [emb_layer(cur_tokens) for cur_tokens, emb_layer in zip(shifted_audio_tokens, self.depth_embeddings, strict=True)],
        1,
    )
    depthformer_tokens[:, -1] *= 0
    depthformer_tokens = depthformer_tokens.roll(1, 1)
    depthformer_in += depthformer_tokens

    if depthformer_in.numel() == 0:
        depthformer_in = rearrange(depthformer_in, "L C D -> C L D")

    should_split = len(depthformer_in) >= 2**14
    k = int(math.log2(len(depthformer_in))) - 14 + 1 if should_split else 0
    num_chunks = 2**k
    depthformer_out = torch.cat([self.depthformer(chunk_in) for chunk_in in depthformer_in.chunk(num_chunks)])

    if depthformer_out.numel() == 0:
        depthformer_out = rearrange(depthformer_out, "C L D -> L C D")

    depthformer_logits = torch.stack(
        [self.depth_embeddings[i].get_logits(depthformer_out[:, i]) for i in range(self.codebooks)]
    )
    audio_logits = rearrange(depthformer_logits, "C L V -> (L C) V")
    shifted_audio_tokens = rearrange(shifted_audio_tokens, "C L -> (L C)")

    return text_logits, audio_logits, shifted_text_tokens, shifted_audio_tokens


def install_vision(
    model: LFM2AudioModel,
    *,
    rows_path: str | Path,
    init_variant: str = "omp",
    vl_repo: str = VL_REPO,
    unfreeze_projector: bool = False,
    verbose: bool = True,
) -> None:
    """Attach the transplained tower+projector, splice the image-token rows
    (embed_tokens.weight is tied to the LM head, so one write fixes both),
    and rebind _prefill/logits on this instance."""
    from safetensors.torch import load_file

    device = next(model.parameters()).device
    vl = Lfm2VlForConditionalGeneration.from_pretrained(vl_repo, dtype=torch.bfloat16)
    model.vision_tower = vl.model.vision_tower.to(device)
    model.multi_modal_projector = vl.model.multi_modal_projector.to(device)
    del vl

    for p in model.vision_tower.parameters():
        p.requires_grad_(False)
    if not unfreeze_projector:
        for p in model.multi_modal_projector.parameters():
            p.requires_grad_(False)

    rows = load_file(str(rows_path))["rows"]
    n = RANGE_HI - RANGE_LO + 1
    assert tuple(rows.shape) == (n, model.lfm.config.hidden_size), rows.shape
    with torch.no_grad():
        model.lfm.embed_tokens.weight[RANGE_LO : RANGE_HI + 1] = rows.to(
            device=device, dtype=model.lfm.embed_tokens.weight.dtype)

    model._vision_features = MethodType(_vision_features, model)
    model._prefill_orig = model._prefill
    model._prefill = MethodType(_vision_prefill, model)
    model.logits_orig = model.logits
    model.logits = MethodType(_vision_logits, model)
    if verbose:
        print(f"vision installed: variant={init_variant} rows[{RANGE_LO}:{RANGE_HI+1}] "
              f"tower+projector from {vl_repo} (projector trainable={unfreeze_projector})",
              flush=True)


# ---------------------------------------------------------------- data path

@dataclass(slots=True, kw_only=True)
class VisionRow:
    text: torch.Tensor
    audio_in: torch.Tensor
    audio_in_lens: torch.Tensor
    audio_out: torch.Tensor
    modality_flag: torch.Tensor
    supervision_mask: torch.Tensor
    pixel_values: torch.Tensor
    spatial_shapes: torch.Tensor
    pixel_attention_mask: torch.Tensor


@dataclass(slots=True, kw_only=True)
class VisionBatch:
    text: torch.Tensor
    audio_in: torch.Tensor
    audio_in_lens: torch.Tensor
    audio_out: torch.Tensor
    modality_flag: torch.Tensor
    supervision_mask: torch.Tensor
    pixel_values: torch.Tensor
    spatial_shapes: torch.Tensor
    pixel_attention_mask: torch.Tensor

    def to(self, device: torch.device | str) -> VisionBatch:
        return VisionBatch(
            text=self.text.to(device), audio_in=self.audio_in.to(device),
            audio_in_lens=self.audio_in_lens.to(device), audio_out=self.audio_out.to(device),
            modality_flag=self.modality_flag.to(device),
            supervision_mask=self.supervision_mask.to(device),
            pixel_values=self.pixel_values.to(device),
            spatial_shapes=self.spatial_shapes.to(device),
            pixel_attention_mask=self.pixel_attention_mask.to(device),
        )


class VisionDataLoader(torch.utils.data.Dataset[VisionRow]):
    """Vision pack rows: {input_ids (with <image> ids), modality_flag,
    supervision_mask, image (PNG)}. The image processor runs here so the
    trainer sees exactly the build-time token layout."""

    def __init__(self, dataset_path: str, image_processor, context_length: int = 1024):
        self.dataset = load_from_disk(dataset_path)
        self.ip = image_processor
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> VisionRow:
        row = self.dataset[idx]
        ids = torch.as_tensor(row["input_ids"], dtype=torch.long)
        modality = torch.as_tensor(row["modality_flag"], dtype=torch.long)
        supervision = torch.as_tensor(row["supervision_mask"], dtype=torch.bool)
        text = ids[modality != IMAGE]

        feats = self.ip(row["image"], return_row_col_info=True)
        pixel_values = torch.as_tensor(feats["pixel_values"])
        spatial_shapes = torch.as_tensor(feats["spatial_shapes"], dtype=torch.long)
        pixel_attention_mask = torch.as_tensor(feats["pixel_attention_mask"], dtype=torch.long)
        n_img = int((modality == IMAGE).sum())
        n_feat = int(pixel_attention_mask.sum()) // 4  # 2x2 merge
        assert n_img == n_feat, f"row {idx}: {n_img} image slots vs {n_feat} features"

        pad = self.context_length - int(modality.shape[0])
        if pad < 0:
            raise ValueError(f"row {idx} len {modality.shape[0]} > context {self.context_length}")
        # No end-padding: batch is always 1, and generation conditions its
        # first token on the last prompt position — padding would sit between
        # the prompt and the generated caption there (training never sees that
        # layout, because supervision ends before the pads under causal attention).
        return VisionRow(
            text=text,
            audio_in=torch.empty((128, 0), dtype=torch.float32),
            audio_in_lens=torch.empty((0,), dtype=torch.long),
            audio_out=torch.empty((8, 0), dtype=torch.long),
            modality_flag=modality,
            supervision_mask=supervision,
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
        )


def vision_collator(batch: list[VisionRow]) -> VisionBatch:
    assert len(batch) == 1, "vision lane runs batch 1 (variable tile counts)"
    r = batch[0]
    return VisionBatch(
        text=r.text.unsqueeze(0), audio_in=r.audio_in, audio_in_lens=r.audio_in_lens,
        audio_out=r.audio_out, modality_flag=r.modality_flag.unsqueeze(0),
        supervision_mask=r.supervision_mask.unsqueeze(0),
        pixel_values=r.pixel_values, spatial_shapes=r.spatial_shapes,
        pixel_attention_mask=r.pixel_attention_mask,
    )


def build_vision_prompt(system: str) -> str:
    """Liquid chat template with an image in the user turn (mirrors
    LFM2AudioChatMapper's layout; the processor expands <image> in place)."""
    return (
        "<|startoftext|>"
        f"<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n<image><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@torch.no_grad()
def generate_text(model, batch: VisionBatch, max_new_tokens: int = 256) -> str:
    """Greedy text-only decode over a VisionBatch prompt (stock
    generate_sequential cannot pass image features through _prefill).

    Pack rows carry prompt AND caption (supervision marks the caption), so the
    prompt for generation is the prefix BEFORE the first supervised position —
    feeding the whole row would ask the model what follows its own <|im_end|>.
    """
    sup = batch.supervision_mask[0]
    first_sup = int(sup.nonzero()[0].item()) if sup.any() else batch.modality_flag.shape[1]
    mf_pre = batch.modality_flag[0, :first_sup]
    text = batch.text[0, : int((mf_pre != IMAGE).sum())].unsqueeze(0)
    feats = model._vision_features(
        batch.pixel_values, batch.spatial_shapes, batch.pixel_attention_mask)
    in_emb = model._prefill(
        text=text, audio_in=batch.audio_in, audio_in_lens=batch.audio_in_lens,
        audio_out=batch.audio_out, modality_flag=mf_pre.unsqueeze(0), image_embeds=feats)
    cache = None
    out: list[int] = []
    for _ in range(max_new_tokens):
        res = model.lfm(inputs_embeds=in_emb, past_key_values=cache, use_cache=True)
        cache = res.past_key_values
        logits = torch.nn.functional.linear(res.last_hidden_state[0, -1], model.lfm.embed_tokens.weight)
        nxt = int(logits.argmax())
        if nxt == 7:  # <|im_end|>
            break
        out.append(nxt)
        in_emb = model.lfm.embed_tokens(torch.tensor([[nxt]], device=in_emb.device))
    return out
