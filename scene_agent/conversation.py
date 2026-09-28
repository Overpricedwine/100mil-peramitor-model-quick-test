"""Conversation-aware local director for improvement turns.

The visual grader and the director are separate roles. Both receive the
rebuilt scene and full saved conversation, while only the director emits the
next user-style instruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .core import SceneSession
from .prompt_templates import CONVERSATION_DIRECTOR_SYSTEM_PROMPT


@dataclass(frozen=True)
class DirectorDecision:
    action: str
    request: str = ""


def build_review_context(
    session: SceneSession, original_goal: str, last_function_id: str | None,
    previous_grade: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Exact data supplied to both grader and conversation director.

    ``model_context`` rebuilds the active scene on every call. It therefore
    includes edits, deletion/restoration effects, all active properties, and
    every persisted conversation turn rather than only the latest function.
    """
    return {
        "original_goal": original_goal,
        "last_function_id": last_function_id,
        "previous_grade": previous_grade,
        "session": session.model_context(),
    }


class LocalQwenConversationDirector:
    """Uses a local Qwen checkpoint to select the next conversational goal."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", max_new_tokens: int = 160) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens = max_new_tokens

    def decide(self, review_context: dict[str, Any], grader_result: dict[str, Any]) -> DirectorDecision:
        import torch

        payload = {**review_context, "grader_result": grader_result}
        messages = [
            {"role": "system", "content": CONVERSATION_DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        response = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        start, end = response.find("{"), response.rfind("}")
        if start < 0 or end < start:
            raise ValueError("conversation director did not return JSON")
        value = json.loads(response[start:end + 1])
        action = value.get("action")
        request = value.get("request", "")
        if action not in {"continue", "finish"} or not isinstance(request, str):
            raise ValueError("invalid conversation director JSON")
        return DirectorDecision(action=action, request=request)
