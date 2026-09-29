"""restraint_v0 variant c: per-token loss weighting on copied (unchanged) target tokens.

`train_lora.py --keep-weight W` routes the trainer through `keep_weight_trainer_cls(W)`, which
returns an SFTTrainer subclass whose compute_loss re-weights the per-token cross entropy: W on
target tokens that sit inside a span copied from the input, 1.0 on tokens the model has to edit.
The loss is the weighted mean, so its scale matches the unweighted run.

Which target tokens are "copies" is decided in token-id space, per row, at each step:
  * the completion tokens are `labels != -100`;
  * the input text is the tail of the prompt (chat_messages puts the raw ASR text in the last
    user turn, so it ends a few template tokens before the completion starts). We take a window
    of `1.4 * len(completion) + 24` prompt tokens back from the completion, which covers the
    user turn and keeps the system prompt and the vocabulary line out of the comparison;
  * difflib over (window, completion) marks every `equal` block of at least MIN_BLOCK tokens;
    those completion positions get weight W.

Nothing here runs unless --keep-weight is passed; the default path is untouched.
"""
from __future__ import annotations

import difflib

MIN_BLOCK = 2
WINDOW_MULT = 1.4
WINDOW_ADD = 24


def keep_mask_row(prompt_ids: list[int], tgt_ids: list[int]) -> list[bool]:
    """True where tgt_ids[j] sits inside a >=MIN_BLOCK token span copied from the input tail."""
    if not tgt_ids or not prompt_ids:
        return [False] * len(tgt_ids)
    win = int(WINDOW_MULT * len(tgt_ids)) + WINDOW_ADD
    a = prompt_ids[-win:]
    keep = [False] * len(tgt_ids)
    sm = difflib.SequenceMatcher(a=a, b=tgt_ids, autojunk=False)
    for i1, j1, size in sm.get_matching_blocks():
        if size >= MIN_BLOCK:
            for j in range(j1, j1 + size):
                keep[j] = True
    return keep


def keep_weight_trainer_cls(weight: float, base_cls=None):
    import torch
    import torch.nn.functional as F

    if base_cls is None:
        from trl import SFTTrainer as base_cls  # noqa: N813

    class KeepWeightSFTTrainer(base_cls):
        keep_weight = float(weight)

        _cache: dict = {}

        def _keep_weights(self, input_ids, labels):
            """(B, T) float tensor of per-token weights aligned with `labels`.

            Rows repeat across epochs, so the mask is cached on the (prompt tail, target)
            token tuple; difflib then runs once per distinct row instead of once per step.
            """
            w = torch.ones_like(labels, dtype=torch.float32)
            ids = input_ids.tolist()
            labs = labels.tolist()
            for b, (row_ids, row_lab) in enumerate(zip(ids, labs)):
                pos = [t for t, v in enumerate(row_lab) if v != -100]
                if not pos:
                    continue
                start = pos[0]
                tgt = [row_lab[t] for t in pos]
                win = int(WINDOW_MULT * len(tgt)) + WINDOW_ADD
                head = tuple(row_ids[max(0, start - win):start])
                key = (head, tuple(tgt))
                keep = self._cache.get(key)
                if keep is None:
                    keep = keep_mask_row(list(head), tgt)
                    if len(self._cache) < 400000:
                        self._cache[key] = keep
                for t, k in zip(pos, keep):
                    if k:
                        w[b, t] = self.keep_weight
            return w

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            fwd = {k: v for k, v in inputs.items()
                   if k in ("input_ids", "attention_mask", "position_ids", "inputs_embeds")}
            outputs = model(**fwd)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            w = self._keep_weights(inputs["input_ids"], labels)[..., 1:].contiguous()
            mask = shift_labels != -100
            safe = shift_labels.masked_fill(~mask, 0)
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                safe.view(-1), reduction="none",
            ).view(safe.shape)
            w = w.to(ce.dtype) * mask.to(ce.dtype)
            denom = w.sum().clamp_min(1e-6)
            loss = (ce * w).sum() / denom
            return (loss, outputs) if return_outputs else loss

    return KeepWeightSFTTrainer
