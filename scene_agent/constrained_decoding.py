"""Incremental guard rails for the model's single JSON scene-function output.

This keeps malformed byte sequences, a second JSON document, and an invalid
top-level close out of sampling.  `grammar.parse_generated_json` remains the
authority for the complete function schema and scene-side semantics.
"""

from __future__ import annotations

from .grammar import parse_generated_json


_OUTSIDE_STRING = set('{}[],:-.+0123456789eEtrufalsn \t\r\n')


def is_json_object_prefix(text: str) -> bool:
    """Whether *text* can still become one JSON object (lexical level)."""
    stripped = text.lstrip()
    if stripped and not stripped.startswith("{"):
        return False

    stack: list[str] = []
    in_string = False
    escaping = False
    unicode_digits = 0
    root_closed = False
    seen_root = False

    for char in text:
        if root_closed:
            if not char.isspace():
                return False
            continue
        if in_string:
            if unicode_digits:
                if char not in "0123456789abcdefABCDEF":
                    return False
                unicode_digits -= 1
                continue
            if escaping:
                if char == "u":
                    unicode_digits = 4
                elif char not in '"\\/bfnrt':
                    return False
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == '"':
                in_string = False
            elif ord(char) < 0x20:
                return False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            if not seen_root:
                if char != "{":
                    return False
                seen_root = True
            stack.append(char)
        elif char in "}]":
            if not stack:
                return False
            opener = stack.pop()
            if (opener, char) not in (("{", "}"), ("[", "]")):
                return False
            if not stack:
                root_closed = True
        elif char not in _OUTSIDE_STRING:
            return False
    return True


class GrammarLogitsProcessor:
    """Masks invalid next BPE tokens during generation of one function.

    The processor deliberately permits incomplete schema values, but disallows
    closing the outer object unless strict JSON/schema validation succeeds.
    That makes every EOS-reachable output a safe function call.
    """

    def __init__(self, tokenizer, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self._pieces = [tokenizer.decode([token_id]) for token_id in range(tokenizer.vocab_size)]
        special_ids = {tokenizer.tokenizer.token_to_id(token) for token in
                       ("<pad>", "<bos>", "<eos>", "<scene>", "</scene>", "<function>")}
        self._special_ids = {token_id for token_id in special_ids if token_id is not None}

    def allowed_token_ids(self, generated_ids: list[int]) -> list[int]:
        text = self.tokenizer.decode(generated_ids)
        try:
            parse_generated_json(text)
        except (ValueError, TypeError):
            complete = False
        else:
            complete = True
        if complete:
            return [self.tokenizer.eos_id]

        allowed: list[int] = []
        for token_id, piece in enumerate(self._pieces):
            if token_id in self._special_ids or not piece:
                continue
            candidate = text + piece
            if not is_json_object_prefix(candidate):
                continue
            # A close that creates a complete but schema-invalid object cannot
            # be repaired by later tokens, so never sample it.
            if candidate.rstrip().endswith("}") and not is_json_object_prefix(candidate + "{"):
                try:
                    parse_generated_json(candidate)
                except (ValueError, TypeError):
                    # It is syntactically closed. Reject unless valid above.
                    continue
            allowed.append(token_id)
        return allowed

    def __call__(self, logits, token_ids):
        if logits.shape[0] != 1:
            raise ValueError("grammar-constrained generation supports batch size 1")
        generated_ids = token_ids[0, self.prompt_length:].tolist()
        allowed = self.allowed_token_ids(generated_ids)
        if not allowed:
            raise ValueError("generation reached a grammar dead end")
        # Import only when this CUDA inference path is actually used.  Prefix
        # validation itself should work on a CPU-only development machine.
        import torch

        constrained = torch.full_like(logits, float("-inf"))
        constrained[:, allowed] = logits[:, allowed]
        return constrained
