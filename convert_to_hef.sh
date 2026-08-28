#!/bin/bash
# Fire & Smoke YOLOv8n -> HEF conversion for Hailo-8L
# Run this in your WSL2 hailoenv (source hailoenv/bin/activate first)
set -e

MODEL=finetuned_best
HW_ARCH=hailo8l

echo "=== Step 1: Parse ONNX with explicit end nodes ==="
hailo parser onnx ${MODEL}.onnx --hw-arch $HW_ARCH \
    --end-node-names \
    /model.22/cv2.0/cv2.0.2/Conv \
    /model.22/cv3.0/cv3.0.2/Conv \
    /model.22/cv2.1/cv2.1.2/Conv \
    /model.22/cv3.1/cv3.1.2/Conv \
    /model.22/cv2.2/cv2.2.2/Conv \
    /model.22/cv3.2/cv3.2.2/Conv \
    --har-path ${MODEL}.har -y

echo ""
echo "!!! STOP HERE !!!"
echo "The parser just renamed the six end nodes to internal identifiers"
echo "(e.g. conv41, conv44...). Find them with:"
echo "  cat ${MODEL}.har | python3 -m json.tool | grep -A 10 output_layers_order"
echo "Then fill those real names into nms_config.json and model_script.alls"
echo "(replacing the <FILL_IN_...> placeholders) BEFORE running the rest of"
echo "this script. Re-run from Step 2 onward once that's done."
echo ""
read -p "Press enter once nms_config.json and model_script.alls are filled in..."

echo "=== Step 2: Build calibration set (edit build_calib_set.py path first) ==="
python3 build_calib_set.py

echo "=== Step 3: Optimize (quantize) ==="
hailo optimize ${MODEL}.har --hw-arch $HW_ARCH \
    --calib-set-path calib_set.npy \
    --model-script model_script.alls \
    --output-har-path ${MODEL}_optimized.har

echo "=== Step 4: Compile to HEF ==="
hailo compiler ${MODEL}_optimized.har --hw-arch $HW_ARCH \
    --model-script model_script.alls --output-dir .

echo "=== Step 5: Validate ==="
hailortcli parse-hef ${MODEL}.hef

echo "Done. Transfer ${MODEL}.hef to the Pi and re-run parse-hef there as a final check."
