"""Run a bundle (``htrbench bundle-export``) on a Modal GPU.

    modal volume put htrbench-data bundles/<run_id>.zip /bundles/<run_id>.zip
    modal run scripts/modal_runner.py --bundle <run_id> --adapter hf --gpu a100
    modal volume get htrbench-data /results/<run_id> bundles/<run_id>.results
    htrbench bundle-import bundles/<run_id>.results

Adapter and GPU of every bundle system are recorded in configs/systems.yaml (params.adapter, params.gpu).

    hf         any transformers image-text-to-text model with a chat template (olmOCR-2, Qwen3-VL, InternVL, Nanonets-OCR2):
               the benchmark prompt, greedy decoding, bf16, one page per call
    churro     stanford-oval/churro-3B with the system and user prompt of its model card and <output> tags
    paddle-vl  PaddleOCR-VL through the official page pipeline (layout detection, then the VLM per element); the text is
               the block contents in the pipeline's reading order. Takes no prompt. GPU: paddle
    ppocr      PP-OCRv5 detection and recognition (lang=de), lines sorted top to bottom. Takes no prompt. GPU: paddle

The run resumes from /vol/results/<run_id>/predictions.jsonl, logs GPU board power per page (energy_wh) and writes
run_info.json, from which the importer takes the hardware.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import modal

APP = "htrbench"
VOL = "htrbench-data"
HF_VOL = "htrbench-hf-cache"

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_VOL, create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.8.0", "torchvision==0.23.0")
    .pip_install("transformers>=4.57,<5", "accelerate>=1.0", "qwen-vl-utils", "pillow", "numpy",
                 "huggingface_hub[hf_transfer]", "sentencepiece", "protobuf", "timm", "einops")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)
# PaddleOCR: the paddle GPU wheel comes from Baidu's index (CUDA 12.6 build); torch is kept for the GPU name / sync calls.
paddle_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1", "ccache")
    .pip_install("paddlepaddle-gpu==3.2.1", index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/")
    .pip_install("paddleocr[doc-parser]>=3.4", "pillow", "numpy", "torch==2.8.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1", "PADDLE_PDX_MODEL_SOURCE": "HuggingFace",
          "FLAGS_use_cuda_managed_memory": "false"})
)

CHURRO_SYSTEM_PROMPT = (
    "You are an expert in diplomatic transcription of historical documents from various languages. Your task is "
    "to extract the full text from a given page. Only output the transcribed text between <output> and </output> tags."
)
CHURRO_USER_PROMPT = (
    "Follow these instructions:\n\n"
    "1. You will be provided with a scanned document page.\n\n"
    "2. Perform transcription on the entirety of the page, converting all visible text into the following format. "
    "Include handwritten and print text, if any. Include tables, captions, headers, main text and all other visible text.\n\n"
    "3. If you encounter any non-text elements, simply skip them without attempting to describe them.\n\n"
    "4. Do not modernize or standardize the text. For example, if the transcription is using \"ſ\" instead of \"s\" "
    "or \"а\" instead of \"a\", keep it that way.\n\n"
    "5. When you come across text in languages other than English, transcribe it as accurately as possible without translation.\n\n"
    "6. Output the OCR result in the following format:\n\n<output>\nextracted text here\n</output>\n\n"
    "Remember, your goal is to accurately transcribe the text from the scanned page as much as possible. Process the "
    "entire page, even if it contains a large amount of text, and provide clear, well-formatted output. Pay attention "
    "to the appropriate reading order and layout of the text."
)


def _strip_output_tags(text: str, tag: str = "output") -> str:
    m = re.match(rf"^\s*<{tag}>\s*(.*?)\s*(</{tag}>\s*)?$", text, re.S)
    if m:
        text = m.group(1)
    return re.sub(rf"</?{tag}\b[^>]*>", "", text, flags=re.I).strip()


class _Power:
    """Samples nvidia-smi board power every 0.5 s; energy_wh = mean W * s / 3600."""

    def __init__(self):
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._t = None
        self.t0 = 0.0

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=2).stdout.strip().splitlines()
                self.samples.append(float(out[0]))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self.t0 = time.perf_counter()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=2)

    @property
    def energy_wh(self) -> float | None:
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples) * (time.perf_counter() - self.t0) / 3600.0


def _load_image(path: Path, max_side: int):
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)
    return im


def _make_hf(model_id: str, adapter: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(model_id)

    def run(image, prompt: str, max_new: int):
        if adapter == "churro":
            msgs = [{"role": "system", "content": [{"type": "text", "text": CHURRO_SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": CHURRO_USER_PROMPT}]}]
        else:
            msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        try:
            inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        except Exception:  # noqa: BLE001 - older processors
            text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inputs = proc(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(model.device)
        n_in = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
        gen = out[0, n_in:]
        raw = proc.decode(gen, skip_special_tokens=True)
        text = _strip_output_tags(raw) if adapter == "churro" else raw.strip()
        return text, n_in, int(gen.shape[0])

    return run


def _paddle_blocks_text(res) -> str:
    """Plain text of a PaddleOCR-VL page result: block contents in the pipeline's reading order, image and chart blocks
    skipped, table HTML reduced to cell text."""
    import re as _re

    j = getattr(res, "json", None) or {}
    r = j.get("res", j) if isinstance(j, dict) else {}
    blocks = r.get("parsing_res_list") or r.get("layout_parsing_result") or []
    skip = {"image", "chart", "seal", "figure", "picture"}
    lines = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        lab = str(b.get("block_label", b.get("label", ""))).lower()
        if lab in skip:
            continue
        content = b.get("block_content", b.get("content", "")) or ""
        if "<" in content and ">" in content:  # table / html -> cell text
            content = _re.sub(r"</t[dh]>", " | ", content)
            content = _re.sub(r"</tr>", "\n", content)
            content = _re.sub(r"<[^>]+>", "", content)
        for ln in str(content).splitlines():
            if ln.strip():
                lines.append(ln.strip())
    if not lines:  # a page the layout step found no blocks on: fall back to the pipeline's markdown
        md = getattr(res, "markdown", None) or {}
        text = md.get("markdown_texts", "") if isinstance(md, dict) else str(md)
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip() and not ln.strip().startswith("![")]
    return "\n".join(lines)


def _make_paddle_vl(model_id: str):
    """PaddleOCR-VL 1.6 through the official page-level pipeline (layout detection + VLM element recognition)."""
    from paddleocr import PaddleOCRVL

    version = "v1.6" if "1.6" in (model_id or "") else "v1.5" if "1.5" in (model_id or "") else "v1.6"
    pipe = PaddleOCRVL(pipeline_version=version, use_doc_orientation_classify=False, use_doc_unwarping=False, use_chart_recognition=False)
    print(f"paddleocr-vl pipeline {version} ready")

    def run(image, prompt: str, max_new: int):
        tmp = "/tmp/htrbench_page.png"
        image.save(tmp)
        out = list(pipe.predict(tmp))
        texts = [_paddle_blocks_text(res) for res in out]
        return "\n".join(t for t in texts if t).strip(), None, None

    return run


def _make_ppocr(model_id: str):
    """PP-OCRv5 (text detection + CTC recognition, lang=de -> latin recogniser): lines in top-to-bottom, left-to-right order."""
    from paddleocr import PaddleOCR

    lang = "de"
    m = re.search(r"lang=([a-z_]+)", model_id or "")
    if m:
        lang = m.group(1)
    ocr = PaddleOCR(lang=lang, ocr_version="PP-OCRv5", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    print(f"pp-ocrv5 ready (lang={lang})")

    def run(image, prompt: str, max_new: int):
        tmp = "/tmp/htrbench_page.png"
        image.save(tmp)
        out = list(ocr.predict(tmp))
        texts = []
        for res in out:
            j = getattr(res, "json", None) or {}
            r = j.get("res", j) if isinstance(j, dict) else {}
            rec = list(r.get("rec_texts") or [])
            polys = r.get("rec_polys") if r.get("rec_polys") is not None else r.get("dt_polys")
            polys = list(polys) if polys is not None else []
            for i, t in enumerate(rec):
                if i < len(polys) and len(polys[i]):
                    ys = [float(p[1]) for p in polys[i]]
                    xs = [float(p[0]) for p in polys[i]]
                    key = (min(ys), min(xs))
                else:
                    key = (float(i), 0.0)
                texts.append((key, str(t)))
        texts.sort(key=lambda kt: kt[0])  # by the top edge, then the left edge, of each detected line
        return "\n".join(t for _, t in texts if t.strip()).strip(), None, None

    return run


def _prompt_note(adapter: str, prompt_id: str) -> str:
    if adapter in ("paddle-vl", "ppocr"):
        return f"benchmark prompt {prompt_id} ignored (the pipeline takes none)"
    if adapter == "churro":
        return f"benchmark prompt {prompt_id} replaced by the system and user prompt of the model card"
    return f"benchmark prompt {prompt_id} passed to the model"


def _run_bundle(bundle: str, adapter: str, model_id: str | None, limit: int | None, max_new_tokens: int, max_side: int) -> dict:
    import torch

    root = Path("/vol/bundles") / bundle
    zpath = Path("/vol/bundles") / f"{bundle}.zip"
    if not (root / "tasks.jsonl").exists():
        if not zpath.exists():
            raise SystemExit(f"bundle zip not on the volume: {zpath} (modal volume put {VOL} bundles/{bundle}.zip /bundles/)")
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root)
        vol.commit()
    meta = json.loads((root / "bundle_meta.json").read_text(encoding="utf-8"))
    tasks = [json.loads(ln) for ln in (root / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    if limit:
        tasks = tasks[:limit]
    model_id = model_id or meta["model_id"]
    out_dir = Path("/vol/results") / bundle
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    done = {json.loads(ln)["page_id"] for ln in pred_path.read_text(encoding="utf-8").splitlines() if ln.strip()} if pred_path.exists() else set()
    todo = [t for t in tasks if t["page_id"] not in done]
    gpu = torch.cuda.get_device_name(0)
    print(f"{bundle}: model={model_id} adapter={adapter} gpu={gpu} pages={len(tasks)} done={len(done)} todo={len(todo)}")
    if not todo:
        return {"bundle": bundle, "done": len(done), "new": 0, "errors": 0}
    t_load = time.perf_counter()
    if adapter == "paddle-vl":
        run = _make_paddle_vl(model_id)
    elif adapter == "ppocr":
        run = _make_ppocr(model_id)
    else:
        run = _make_hf(model_id, adapter)
    print(f"model loaded in {time.perf_counter() - t_load:.0f}s; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t_all = time.perf_counter()
    n_err = 0
    with pred_path.open("a", encoding="utf-8") as fh:
        for i, t in enumerate(todo):
            row = {"page_id": t["page_id"]}
            t0 = time.perf_counter()
            try:
                with _Power() as ps:
                    text, tin, tout = run(_load_image(root / t["image"], max_side), t["prompt"], max_new_tokens)
                    torch.cuda.synchronize()
                row.update(text=text, tokens_in=tin, tokens_out=tout, error=None, energy_wh=ps.energy_wh)
            except Exception as e:  # noqa: BLE001
                n_err += 1
                row.update(text="", error=str(e)[:300])
                torch.cuda.empty_cache()
            row["latency_s"] = time.perf_counter() - t0
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(todo) - 1:
                vol.commit()
                preview = (row.get("text") or row.get("error") or "")[:60].replace("\n", " / ")
                print(f"{i + 1}/{len(todo)} {t['page_id']} {row['latency_s']:.1f}s {preview}")
    info = {"gpu": gpu, "platform": "modal", "torch_version": torch.__version__, "model_revision": model_id, "started_at": started,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "total_wall_s": time.perf_counter() - t_all,
            "notes": f"adapter={adapter}; {_prompt_note(adapter, meta.get('prompt_id'))}; max_side={max_side}; "
                     f"max_new_tokens={max_new_tokens}; greedy; bf16"}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    shutil.copy(root / "bundle_meta.json", out_dir / "bundle_meta.json")
    vol.commit()
    return {"bundle": bundle, "done": len(done), "new": len(todo), "errors": n_err, "wall_s": round(info["total_wall_s"])}


_COMMON = dict(volumes={"/vol": vol, "/hf": hf_cache}, timeout=4 * 3600, retries=0)


@app.function(image=base_image, gpu="A10G", **_COMMON)
def run_a10g(bundle: str, adapter: str = "hf", model_id: str | None = None, limit: int | None = None, max_new_tokens: int = 2048,
             max_side: int = 1800) -> dict:
    return _run_bundle(bundle, adapter, model_id, limit, max_new_tokens, max_side)


@app.function(image=base_image, gpu="A100-80GB", **_COMMON)
def run_a100(bundle: str, adapter: str = "hf", model_id: str | None = None, limit: int | None = None, max_new_tokens: int = 2048,
             max_side: int = 1800) -> dict:
    return _run_bundle(bundle, adapter, model_id, limit, max_new_tokens, max_side)


@app.function(image=base_image, gpu="H100", **_COMMON)
def run_h100(bundle: str, adapter: str = "hf", model_id: str | None = None, limit: int | None = None, max_new_tokens: int = 2048,
             max_side: int = 1800) -> dict:
    return _run_bundle(bundle, adapter, model_id, limit, max_new_tokens, max_side)


@app.function(image=paddle_image, gpu="A10G", **_COMMON)
def run_paddle(bundle: str, adapter: str = "paddle-vl", model_id: str | None = None, limit: int | None = None, max_new_tokens: int = 2048,
               max_side: int = 1800) -> dict:
    return _run_bundle(bundle, adapter, model_id, limit, max_new_tokens, max_side)


@app.local_entrypoint()
def main(bundle: str, adapter: str = "hf", gpu: str = "a100", model_id: str = "", limit: int = 0, max_new_tokens: int = 2048,
         max_side: int = 1800):
    key = "paddle" if adapter in ("paddle-vl", "ppocr") else gpu
    fn = {"a10g": run_a10g, "a100": run_a100, "h100": run_h100, "paddle": run_paddle}[key]
    print(fn.remote(bundle, adapter=adapter, model_id=model_id or None, limit=limit or None, max_new_tokens=max_new_tokens, max_side=max_side))
