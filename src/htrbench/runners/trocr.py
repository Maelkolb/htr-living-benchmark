"""TrOCR line recogniser (VisionEncoderDecoderModel), fp16 on CUDA, batched beam search."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from ..schema import SystemSpec
from .base import Runner
from .linepage import LineModelMixin

log = logging.getLogger("htrbench.trocr")

_ROBERTA_SPECIALS = {"bos_token": "<s>", "eos_token": "</s>", "sep_token": "</s>", "cls_token": "<s>",
                     "pad_token": "<pad>", "unk_token": "<unk>", "mask_token": "<mask>"}


def _fast_tokenizer_from_bpe_files(d: Path):
    """The Kurrent checkpoints ship vocab.json and merges.txt but no tokenizer.json, which transformers 5 can no longer
    convert. The fast tokenizer is therefore built directly with ``tokenizers``."""
    import json

    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    specials = dict(_ROBERTA_SPECIALS)
    stm = d / "special_tokens_map.json"
    if stm.exists():
        for k, v in json.loads(stm.read_text(encoding="utf-8")).items():
            if k in specials:
                specials[k] = v["content"] if isinstance(v, dict) else v
    tk = ByteLevelBPETokenizer(str(d / "vocab.json"), str(d / "merges.txt"), add_prefix_space=False)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, **specials)
    tok.add_special_tokens({k: v for k, v in specials.items()})  # register ids for skip_special_tokens
    return tok


def load_processor(mid: str):
    """The checkpoint's own processor, or one built from its BPE files. Never another model's: a substituted
    tokenizer would turn into a reading, and the benchmark would score it as one."""
    from transformers import AutoImageProcessor, TrOCRProcessor

    try:
        return TrOCRProcessor.from_pretrained(mid)
    except Exception as e:  # noqa: BLE001
        log.warning("TrOCRProcessor.from_pretrained(%s) failed: %s", mid, str(e).splitlines()[0][:160])
    d = Path(mid)
    if not d.is_dir():
        from huggingface_hub import snapshot_download

        d = Path(snapshot_download(mid, allow_patterns=["*.json", "*.txt"]))
    return TrOCRProcessor(image_processor=AutoImageProcessor.from_pretrained(d), tokenizer=_fast_tokenizer_from_bpe_files(d))


class TrOCRRunner(LineModelMixin, Runner):
    def __init__(self, spec: SystemSpec, params: dict | None = None):
        super().__init__(spec, params)
        self.model = None
        self.processor = None
        self.dtype = None

    def setup(self) -> None:
        import torch
        from transformers import VisionEncoderDecoderModel

        mid = self.spec.model_id
        if Path(mid).exists():
            mid = Path(mid).as_posix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        use_fp16 = bool(self.params.get("fp16", True)) and self.device == "cuda"
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.processor = load_processor(mid)
        self.model = VisionEncoderDecoderModel.from_pretrained(mid, dtype=self.dtype).to(self.device).eval()
        log.info("loaded %s on %s (%s)", mid, self.device, "float16" if use_fp16 else "float32")

    def teardown(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def recognize_images(self, paths: list[Path]) -> list[str]:
        import torch
        from PIL import Image

        if self.model is None:
            self.setup()
        bs = int(self.params.get("batch_size", 16))
        num_beams = int(self.params.get("num_beams", 4))
        max_new = int(self.params.get("max_new_tokens", 96))
        out: list[str] = []
        for i in range(0, len(paths), bs):
            ims = [Image.open(p).convert("RGB") for p in paths[i : i + bs]]
            pv = self.processor(images=ims, return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)
            with torch.inference_mode():
                ids = self.model.generate(pv, num_beams=num_beams, max_new_tokens=max_new)
            out.extend(s.strip() for s in self.processor.batch_decode(ids, skip_special_tokens=True))
            for im in ims:
                im.close()
        return out
