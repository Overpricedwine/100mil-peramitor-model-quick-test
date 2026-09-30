"""Incremental guard rails for the model's single JSON scene-function output.

This keeps malformed byte sequences, a second JSON document, and an invalid
top-level close out of sampling.  `grammar.parse_generated_json` remains the
authority for the complete function schema and scene-side semantics.
"""

from __future__ import annotations

from .grammar import parse_generated_json


_OUTSIDE_STRING = set('{}[],:-.+0123456789eEtrufalsn \t\r\n')
_OPERATION_PREFIX = '{"op":"'
_CANONICAL_OPERATIONS = (
    "add_part", "add_rig_root", "add_rig_part", "set_part_properties",
    "delete_function", "restore_function", "finish",
)


def is_canonical_operation_prefix(text: str) -> bool:
    """Require the first JSON member to select a supported function.

    The model is never allowed to invent an operation name.  It may choose one
    of the prebuilt functions, but only after producing the fixed ``{"op":``
    envelope.  Once the operation's closing quote is present, ordinary JSON
    prefix validation handles that function's arguments; the final validator
    remains the authority for its complete schema.
    """
    if not text.startswith(_OPERATION_PREFIX):
        return _OPERATION_PREFIX.startswith(text)
    remainder = text[len(_OPERATION_PREFIX):]
    return any(
        operation.startswith(remainder) or remainder.startswith(operation + '"')
        for operation in _CANONICAL_OPERATIONS
    )


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

    def allowed_token_ids(self, generated_ids: list[int], candidate_ids: list[int] | None = None) -> list[int]:
        """Return grammar-safe continuations.

        ``candidate_ids`` lets the CUDA decoder test only its most likely
        tokens first.  The previous implementation evaluated every BPE token
        (thousands of Python JSON-prefix checks) on every generated token,
        making practical constrained inference far slower than the model.
        We retain a complete-vocabulary fallback in ``__call__``, so this is a
        performance change rather than a relaxation of the safety boundary.
        """
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
        token_ids = candidate_ids if candidate_ids is not None else range(len(self._pieces))
        for token_id in token_ids:
            piece = self._pieces[token_id]
            if token_id in self._special_ids or not piece:
                continue
            # JSON may have harmless leading whitespace in general, but this
            # function language has no reason to generate it.  Requiring the
            # opening brace at token zero prevents a weak checkpoint from
            # consuming its entire function budget with spaces/newlines.
            if not text and not piece.startswith("{"):
                continue
            candidate = text + piece
            if not is_canonical_operation_prefix(candidate):
                continue
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
        # The model's output layer has 16,384 entries while the compact BPE
        # vocabulary is smaller.  First ask its top 512 *representable* tokens
        # and only scan all BPE entries if none can continue valid JSON.
        import torch

        usable = logits[:, :len(self._pieces)]
        candidate_count = min(512, usable.shape[-1])
        candidates = torch.topk(usable, candidate_count, dim=-1).indices[0].tolist()
        allowed = self.allowed_token_ids(generated_ids, candidates)
        if not allowed:
            allowed = self.allowed_token_ids(generated_ids)
        if not allowed:
            raise ValueError("generation reached a grammar dead end")
        # Import only when this CUDA inference path is actually used.  Prefix
        # validation itself should work on a CPU-only development machine.
        constrained = torch.full_like(logits, float("-inf"))
        constrained[:, allowed] = logits[:, allowed]
        return constrained
