"""Local request generation for verified supervised training examples."""

from __future__ import annotations

import json
from typing import Any

from .prompt_templates import REQUEST_AUTHOR_SYSTEM_PROMPT


class LocalQwenRequestAuthor:
    """Runs the local Qwen model in *request-author* mode, not grader mode.

    The deterministic scene specification remains the source of truth. Random
    seeds only produce wording diversity; they never choose the target scene
    functions.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", max_new_tokens: int = 160) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens = max_new_tokens

    def write(self, scene_spec: dict[str, Any], detail_level: str, variation_seed: int, style_hint: str) -> str:
        if detail_level not in {"minimal", "specific"}:
            raise ValueError("detail_level must be minimal or specific")
        payload = {
            "scene_spec": scene_spec,
            "detail_level": detail_level,
            "variation_seed": variation_seed,
            "style_hint": style_hint,
        }
        messages = [
            {"role": "system", "content": REQUEST_AUTHOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)
        generator = self.torch.Generator(device=self.model.device)
        generator.manual_seed(variation_seed)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
                temperature=0.85, top_p=0.92, generator=generator,
            )
        response = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        start, end = response.find("{"), response.rfind("}")
        if start < 0 or end < start:
            raise ValueError("request author did not return JSON")
        value = json.loads(response[start:end + 1])
        request = value.get("request")
        if not isinstance(request, str) or not request.strip():
            raise ValueError("request author JSON requires a non-empty request")
        return request.strip()
