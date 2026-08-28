# Fire & Smoke Detection — Hailo-8L Deployment

Fire/smoke detection model (YOLOv8n, 2 classes) trained on merged D-Fire + FireAndSmoke datasets, converted to `.hef` and deployed on Raspberry Pi 5 + Hailo-8L AI accelerator.

## Status: Deployed and validated on-device

The model has been trained, converted to Hailo's `.hef` format, deployed to a Raspberry Pi 5, and validated with real inference on known test videos and false-positive stress tests. Results and known limitations are summarized below.

---

## 1. Dataset

- **Base**: [D-Fire](https://www.kaggle.com/datasets/zuriyatfatima/dfire-dataset) (class 0 = smoke, 1 = fire). 15,499 train / 1,722 val / 4,306 test images.
- **Merged in**: FireAndSmoke dataset (Catargiu, Cleju, Ciocoiu — *Sensors* 2024), ~22,970 images, originally 3-class (fire/smoke/other), converted to 2-class to match D-Fire.
- **Merged train set**: 32,804 images. Validation/test sets kept as pure D-Fire only, for a clean, consistent benchmark.

## 2. Model & Training

Two checkpoints were trained; the **fine-tuned (merged-dataset) checkpoint is the one deployed**.

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Smoke | 0.841 | 0.774 | **0.832** | 0.521 |
| Fire | 0.753 | 0.622 | **0.707** | 0.369 |
| **Overall** | 0.797 | 0.698 | **0.769** | 0.445 |

**Known tradeoff**: the fine-tuned checkpoint substantially improved smoke detection, but overall mAP50 (0.769) is slightly below a D-Fire-only baseline (0.788), with fire recall being the weaker metric. The fine-tuned checkpoint was chosen for deployment based on strong real-world video performance (below), but this tradeoff is worth revisiting if fire misses prove costly in practice.

## 3. Real-World Pre-Deployment Verification

The trained checkpoint (pre-conversion, FP32) was run on all frames of 6 unseen 4K test videos:

| Video | Frames | Hit rate (conf=0.25) | Avg smoke conf | Avg fire conf |
|---|---|---|---|---|
| Smoke test 1 | 139 | 100% | 0.41 | — |
| Fire test 1 | 590 | 97% | 0.40 | 0.50 |
| Smoke test 2 | 270 | 100% | 0.75 | — |
| Smoke test 3 | 320 | 100% | 0.44 | — |
| Fire+smoke test | 561 | 100% | 0.77 | 0.44 |
| Smoke test 4 | 712 | 99% | 0.71 | — |

At the more realistic **conf=0.5** deployment threshold, one video's hit rate dropped from 100% to 46%, indicating some real frames sit close to the threshold. This is expected behavior given the model's known precision/recall balance, and was re-verified post-conversion on-device (see below).

## 4. Hailo Conversion (`.hef`)

- **Architecture**: YOLOv8n, 2 classes, ONNX opset 11, imgsz 640, `simplify=True`.
- **Target**: `hailo8l`.
- **Compilation**: fell back to multi-context (3 contexts) — single-context compilation was not feasible for this network on this hardware target. This adds some context-switch overhead vs. single-context, but on-device benchmarks (below) show this is not a practical bottleneck.
- **Quantization**: INT8, calibrated on 1024 real training images. Note: run without GPU acceleration (`optimization level 0`), so the usual bias-correction/fine-tuning quantization steps were skipped — a caveat on the FP32 → INT8 accuracy match, which is why on-device revalidation (Section 5) matters.
- Key config files (`nms_config.json`, `model_script.alls`) are included in this repo. The compiled `.hef`, `.har` intermediate files, and calibration set are **not** committed here due to size — available on request.

## 5. On-Device Validation (Raspberry Pi 5 + Hailo-8L)

### Raw inference benchmark
`hailortcli run finetuned_best.hef` → **107.11 FPS**, well above real-time video requirements.

### Real inference on known test video (fire+smoke combined)
Ran via `hailo-detect-simple` with the compiled `.hef`:

- **At conf=0.3**: both classes detected reliably in nearly every frame, confidence trending upward through the clip (smoke ~0.6→0.8, fire ~0.3→0.75) — consistent with pre-conversion FP32 results.
- **At conf=0.5** (realistic deployment threshold): smoke detection remained strong and consistent (0.7–0.8 through most of the clip). Fire detection was noticeably patchier, dropping out for stretches in the first half of the video — consistent with fire being the weaker class in the FP32 validation metrics above. This is a reproduction of an already-known model limitation, not a new conversion-induced regression.

### False-positive stress test
Three images tested at conf=0.5 to check for false triggers on fire-colored-but-not-fire content:

| Test image | Result |
|---|---|
| Orange sunset sky (moderate) | ✅ Clean — no detections |
| Orange sunset sky (very bright/saturated) | ✅ Clean — no detections |
| Campfire glow (night) | ⚠️ **False positive** — `fire` detected at 0.52 confidence, consistently across frames |

**Known limitation**: the model can false-positive on small, isolated, glowing flame sources (e.g. a legitimate campfire) at the current deployment threshold. It correctly ignores broad fire-colored lighting (sunsets), so this appears specific to compact flame-like shapes rather than a general color-based false trigger. Whether this matters depends on deployment context — cameras near legitimate open flames (candles, stoves, fire pits) may see occasional false alerts. Mitigation options: raise the fire-class-specific threshold, add more "controlled small flame, no smoke" negative examples in a future fine-tune, or require human confirmation before any automated action.

---

## Repo Contents

- `build_calib_set.py` — builds the INT8 calibration set from training images
- `convert_to_hef.sh` — end-to-end ONNX → HEF conversion script
- `nms_config.json` — NMS post-processing config (bbox decoders, thresholds, per-stride layer mapping)
- `model_script.alls` — Hailo model script (normalization, activation functions, NMS postprocess call)

**Not included** (available on request due to size): trained checkpoint (`.pt`), ONNX export, `.har` intermediate files, compiled `.hef`, calibration set (`.npy`, ~1.2GB), test videos.

## Next Steps

- Decide on handling for the campfire false-positive case before broader deployment
- Run the pending industrial smoke/steam stress test (not yet completed)
- Consider whether the fire-recall tradeoff (Section 2) warrants further fine-tuning
