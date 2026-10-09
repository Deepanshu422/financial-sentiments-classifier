import re
from typing import Dict, List, Union
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from src.config import FINAL_MODEL_DIR, ID2LABEL, MAX_LENGTH, BASE_MODEL_NAME


class FinancialSentimentPredictor:
    """
    Ultra-lightweight ONNX inference engine optimized for memory-constrained environments (<512MB RAM).
    """

    def __init__(self, model_dir: str = str(FINAL_MODEL_DIR)):
        quant_model_path = f"{model_dir}/onnx/model_quantized.onnx"
        print(f"[INFO] Initializing ONNX runtime session: {quant_model_path}...")

        # fast tokenizer 
        try: 
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        except Exception:
            # fallback for tokenizer if local files are missing in CI
            print(f"[WARN] Local tokenizer not found at {model_dir}. Falling back to {BASE_MODEL_NAME}...")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)            

        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(
            quant_model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def predict_sentence(self, sentence: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        clean_text = sentence.strip()
        if not clean_text:
            return {
                "sentence": clean_text,
                "sentiment": "neutral",
                "confidence": 0.0,
                "probabilities": {name: 0.0 for name in ID2LABEL.values()},
            }

        inputs = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        logits = self.session.run(["logits"], ort_inputs)[0]
        probs = self._softmax(logits)[0]

        predicted_id = int(np.argmax(probs))
        predicted_label = ID2LABEL[predicted_id]
        confidence = float(probs[predicted_id])

        prob_dict = {
            ID2LABEL[idx]: round(float(prob), 4)
            for idx, prob in enumerate(probs)
        }

        return {
            "sentence": clean_text,
            "sentiment": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
        }

    def predict_long_text(self, text: str) -> Dict[str, Union[str, float, List[Dict]]]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        if not sentences:
            return {
                "overall_sentiment": "neutral",
                "overall_confidence": 0.0,
                "document_probabilities": {name: 0.0 for name in ID2LABEL.values()},
                "sentence_count": 0,
                "sentence_breakdown": [],
            }

        sentence_results = []
        weighted_prob_sums = {name: 0.0 for name in ID2LABEL.values()}
        total_word_count = 0

        for s in sentences:
            pred = self.predict_sentence(s)
            sentence_results.append(pred)

            weight = max(len(s.split()), 1)
            total_word_count += weight

            for label, prob in pred["probabilities"].items():
                weighted_prob_sums[label] += weight * prob

        document_probabilities = {
            label: round(weighted_sum / total_word_count, 4)
            for label, weighted_sum in weighted_prob_sums.items()
        }

        overall_sentiment = max(document_probabilities, key=document_probabilities.get)
        overall_confidence = document_probabilities[overall_sentiment]

        return {
            "overall_sentiment": overall_sentiment,
            "overall_confidence": overall_confidence,
            "document_probabilities": document_probabilities,
            "sentence_count": len(sentences),
            "sentence_breakdown": sentence_results,
        }


_predictor = None

def get_predictor() -> FinancialSentimentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FinancialSentimentPredictor()
    return _predictor