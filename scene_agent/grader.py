from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import torch

from .core import SceneGraph
from .prompt_templates import GRADER_SYSTEM_PROMPT


@dataclass
class Grade:
    score: float
    feedback: list[str]
    repairs: list[dict]


class SceneGrader(Protocol):
    def grade(self, request: str, graph: SceneGraph, renders: dict[str, torch.Tensor]) -> Grade: ...


class HeuristicGrader:
    """Runnable non-AI fallback. Replace with a CUDA VLM grader in production."""

    def grade(self, request: str, graph: SceneGraph, renders: dict[str, torch.Tensor]) -> Grade:
        score = min(55.0, len(graph.parts) * 15.0)
        feedback: list[str] = []
        request_words = {word.strip(".,!? ").lower() for word in request.split()}
        names = " ".join(part.name.lower() for part in graph.parts)
        for word in ("tower", "base", "door", "roof"):
            if word in request_words:
                if word in names:
                    score += 10
                else:
                    feedback.append(f"Requested element appears missing: {word}.")
        if graph.skipped:
            score -= 20
            feedback.append("Some functions are inactive or have unmet dependencies.")
        if not graph.parts:
            feedback.append("Scene is empty.")
        return Grade(score=max(0.0, min(100.0, score)), feedback=feedback, repairs=[])


class LocalQwenVLGrader:
    """A fully local CUDA grader for a Qwen2.5-VL checkpoint.

    Imports are delayed so persistent scene-memory tools work before VLM
    dependencies are installed. Model weights are loaded from a local Hugging
    Face cache after the first download; no image or request is sent to an API.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", max_new_tokens: int = 350) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.max_new_tokens = max_new_tokens
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    @staticmethod
    def _to_pil(image: torch.Tensor):
        from PIL import Image

        pixels = image.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels, mode="RGB")

    @staticmethod
    def _parse_json(text: str) -> dict:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"grader did not return JSON: {text[:300]}")
        value = json.loads(text[start:end + 1])
        if not isinstance(value.get("score"), (int, float)):
            raise ValueError("grader JSON requires numeric score")
        return value

    def grade(self, request: str, graph: SceneGraph, renders: dict[str, torch.Tensor]) -> Grade:
        images = [self._to_pil(renders[name]) for name in ("front", "side", "top")]
        instruction = f"""{GRADER_SYSTEM_PROMPT}
Original user request: {request}
Scene metadata: {json.dumps(graph.metadata())}
The three images are front, side, and top orthographic renders of the same scene."""
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        value = self._parse_json(response)
        return Grade(
            score=float(max(0, min(100, value["score"]))),
            feedback=[str(item) for item in value.get("feedback", [])],
            repairs=[item for item in value.get("repairs", []) if isinstance(item, dict)],
        )
