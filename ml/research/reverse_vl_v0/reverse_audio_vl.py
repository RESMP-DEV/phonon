#!/usr/bin/env python3
"""Reverse LFM2.5 graft: audio input head on the larger VL language model.

The audio model's conformer and audio_adapter are attached to
Lfm2VlForConditionalGeneration. Audio frames become continuous embeddings and
replace a reserved placeholder token in the VL language stream. Vision remains
intact and unused in this lane.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from liquid_audio import LFM2AudioDetokenizer, LFM2AudioModel, LFM2AudioProcessor
from liquid_audio.utils import LFMModality
from transformers import AutoTokenizer, Lfm2VlForConditionalGeneration

# Both tokenizers share this unused reserved slot. It never collides with the
# VL image grid tokens 396-500 or the audio modality tokens 128-132.
AUDIO_PLACEHOLDER_ID = 14
AUDIO_REPO = "LiquidAI/LFM2.5-Audio-1.5B"
VL_REPO = "LiquidAI/LFM2.5-VL-1.6B"
SYSTEM_NAME = "prose_dictation_v1"

LORA_TARGETS = (
    r"^model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|out_proj)"
    r"|conv\.(in_proj|out_proj)|feed_forward\.(w1|w2|w3))$"
)


def value(batch: Any, name: str) -> Any:
    try:
        return batch[name]
    except (TypeError, KeyError, IndexError):
        return getattr(batch, name)


@dataclass
class ReverseAudioVL:
    vl: Lfm2VlForConditionalGeneration
    audio: LFM2AudioModel
    tokenizer: Any
    device: torch.device

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        audio_repo: str = AUDIO_REPO,
        vl_repo: str = VL_REPO,
    ) -> ReverseAudioVL:
        if isinstance(device, str):
            device = torch.device(device)
        # The processor hardcodes cuda() for audio output. This lane never
        # generates audio, so the no-op is safe on CPU/MPS as well.
        LFM2AudioDetokenizer.cuda = lambda self, device=None: self
        audio = LFM2AudioModel.from_pretrained(audio_repo, dtype=dtype, device=device)
        vl = Lfm2VlForConditionalGeneration.from_pretrained(
            vl_repo, dtype=dtype
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(vl_repo)
        return cls(vl=vl, audio=audio, tokenizer=tokenizer, device=device)

    def audio_embeddings(
        self, audio_in: torch.Tensor, audio_in_lens: torch.Tensor
    ) -> torch.Tensor:
        """Run the transplanted conformer and audio adapter."""

        conformer_dtype = next(self.audio.conformer.parameters()).dtype
        audio_in = audio_in.to(device=self.device, dtype=conformer_dtype)
        chunks = audio_in.mT.split(audio_in_lens.tolist())
        if chunks:
            padded = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True)
        else:
            padded = audio_in.new_empty((0, 8 + 1, 128))
        encoded, lengths = self.audio.conformer(
            padded.mT.to(self.device), audio_in_lens.to(self.device)
        )
        mask = (
            torch.arange(encoded.shape[-1], device=encoded.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        return self.audio.audio_adapter(encoded.mT[mask])

    def full_ids(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Expand compact Audio text/audio slots into a VL token sequence."""

        modality = value(batch, "modality_flag").to(self.device)
        ids = torch.full(
            modality.shape,
            AUDIO_PLACEHOLDER_ID,
            dtype=torch.long,
            device=self.device,
        )
        text_mask = modality == int(LFMModality.TEXT)
        ids[text_mask] = value(batch, "text").to(self.device).flatten()
        return ids

    def inputs_embeds(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ids = self.full_ids(batch)
        embeds = self.vl.model.language_model.get_input_embeddings()(ids)
        audio = self.audio_embeddings(value(batch, "audio_in"), value(batch, "audio_in_lens"))
        audio_mask = value(batch, "modality_flag").to(self.device) == int(LFMModality.AUDIO_IN)
        expected = int(audio_mask.sum())
        assert audio.shape[0] == expected, (audio.shape, expected)
        embeds[audio_mask] = audio.to(embeds.dtype)
        return embeds

    def hidden_states(
        self, batch: dict[str, torch.Tensor], *, use_cache: bool = False
    ) -> torch.Tensor:
        embeds = self.inputs_embeds(batch)
        output = self.vl.model.language_model(
            inputs_embeds=embeds, use_cache=use_cache
        )
        return output.last_hidden_state

    def logits(
        self, batch: dict[str, torch.Tensor], *, use_cache: bool = False
    ) -> torch.Tensor:
        return self.vl.lm_head(self.hidden_states(batch, use_cache=use_cache))

    @torch.no_grad()
    def generate(
        self,
        wave: torch.Tensor,
        sampling_rate: int,
        *,
        max_new_tokens: int = 96,
        system: str | None = None,
        prompt_id: str = SYSTEM_NAME,
    ) -> str:
        from liquid_audio import ChatState

        if system is None:
            from prompts import get_prompt

            system = get_prompt(prompt_id)
        proc = LFM2AudioProcessor.from_pretrained(AUDIO_REPO, device=self.device).eval()
        chat = ChatState(proc)
        chat.new_turn("system")
        chat.add_text(system)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(wave.to(self.device), sampling_rate)
        chat.end_turn()
        chat.new_turn("assistant")

        batch = dict(chat)
        embeds = self.inputs_embeds(batch)
        output = self.vl.model.language_model(
            inputs_embeds=embeds, use_cache=True
        )
        cache = output.past_key_values
        tokens: list[int] = []
        for _ in range(max_new_tokens):
            token = int(torch.argmax(self.vl.lm_head(output.last_hidden_state[:, -1, :])))
            if token == 7:  # <|im_end|>
                break
            tokens.append(token)
            next_embed = self.vl.model.language_model.get_input_embeddings()(
                torch.tensor([[token]], device=self.device)
            )
            output = self.vl.model.language_model(
                inputs_embeds=next_embed,
                past_key_values=cache,
                use_cache=True,
            )
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--prompt-id", default=SYSTEM_NAME)
    args = parser.parse_args()
    model = ReverseAudioVL.from_pretrained(device=args.device)
    wav, sr = sf.read(args.audio, dtype="float32")
    wave = torch.from_numpy(wav)
    if wave.dim() == 1:
        wave = wave[None, :]
    text = model.generate(
        wave, int(sr), max_new_tokens=args.max_new_tokens, prompt_id=args.prompt_id
    )
    # Synthetic probes may print text; real Aqua callers must aggregate off-site.
    print(json.dumps({"chars": len(text), "text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
