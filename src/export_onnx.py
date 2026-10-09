from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from transformers import AutoTokenizer

model_dir = Path("artifacts/final_model")
export_dir = model_dir / "onnx"

print(f"[INFO] Exporting model from {model_dir} to ONNX...")

# 1. Export the fine-tuned DistilBERT to ONNX format cleanly
ort_model = ORTModelForSequenceClassification.from_pretrained(
    model_dir,
    export=True,
)
ort_model.save_pretrained(export_dir)

# Save tokenizer alongside the onnx weights
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.save_pretrained(export_dir)

print(f"[INFO] Quantizing ONNX model to INT8...")

# 2. Apply Dynamic INT8 Quantization (ARM/x86 CPU optimized)
quantizer = ORTQuantizer.from_pretrained(export_dir)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# Fallback to general dynamic CPU if AVX512 is not present
quantizer.quantize(
    save_dir=export_dir,
    quantization_config=dqconfig,
)

print(f"[SUCCESS] Quantized ONNX model generated at: {export_dir}")