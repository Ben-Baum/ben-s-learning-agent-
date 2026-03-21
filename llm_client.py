from typing import Type, TypeVar
import json
import os

from openai import OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

# ─── Gemini API via OpenAI-compatible endpoint ────────────────────────────────
# Set your Google API key:
#   export GOOGLE_API_KEY="AIza..."
#
# Model used: gemini-2.5-flash (strong personality + Hebrew + free tier)
# Docs: https://ai.google.dev/gemini-api/docs/openai

_PRIMARY_KEY = os.environ.get("GOOGLE_API_KEY")

def _make_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

_primary_client = _make_client(_PRIMARY_KEY) if _PRIMARY_KEY else None


def _call_with_fallback(fn_primary, fn_backup):
    """Call primary client. Raises clear error if key is missing or rate-limited."""
    if not _primary_client:
        raise RuntimeError(
            "GOOGLE_API_KEY not set.\n"
            "Set it with: export GOOGLE_API_KEY='AIza...'\n"
            "Get a free key at: https://aistudio.google.com"
        )
    try:
        result = fn_primary(_primary_client)
        return result
    except RateLimitError:
        raise RuntimeError(
            "Gemini API rate limit reached.\n"
            "Free tier: 1,000 requests/day. Upgrade at: https://aistudio.google.com"
        )


T = TypeVar("T", bound=BaseModel)


def _strip_markdown_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
    return text.strip()


def _is_schema_not_data(parsed: dict) -> bool:
    """Detect if the LLM returned the JSON schema definition instead of actual data."""
    return "$defs" in parsed or (
        parsed.get("type") == "object" and "properties" in parsed
    )


def _build_field_prompt(model: Type[BaseModel]) -> str:
    """
    Build a human-readable field list from a Pydantic model.
    Much clearer for LLMs than raw JSON Schema — avoids model returning the schema itself.
    """
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})

    def resolve(ref: str) -> dict:
        name = ref.split("/")[-1]
        return defs.get(name, {})

    def describe_type(prop: dict, indent: int = 0) -> str:
        pad = "  " * indent
        if "$ref" in prop:
            sub = resolve(prop["$ref"])
            return describe_obj(sub, indent)
        t = prop.get("type", "")
        if t == "array":
            items = prop.get("items", {})
            inner = describe_type(items, indent + 1)
            return f"array of:\n{pad}  {inner}"
        if t == "object" or "properties" in prop:
            return describe_obj(prop, indent)
        enum = prop.get("enum") or prop.get("anyOf")
        if enum:
            vals = [str(e) if isinstance(e, str) else str(e.get("const", "")) for e in enum]
            return f"{t or 'string'} (one of: {', '.join(v for v in vals if v)})"
        return t or "string"

    def describe_obj(obj: dict, indent: int = 0) -> str:
        pad = "  " * indent
        props = obj.get("properties", {})
        req = set(obj.get("required", []))
        lines = []
        for name, prop in props.items():
            optional = "" if name in req else " (optional)"
            lines.append(f"{pad}{name}{optional}: {describe_type(prop, indent)}")
        return "\n".join(lines)

    return describe_obj(schema)


def call_llm_json(
    *,
    model: str,
    system_prompt: str,
    user_content: str,
    response_model: Type[T],
    max_retries: int = 3,
) -> T:
    field_description = _build_field_prompt(response_model)
    full_system = (
        f"{system_prompt}\n\n"
        f"IMPORTANT: You MUST respond with a single JSON object containing ACTUAL DATA.\n"
        f"Do NOT return a schema, type definitions, or '$defs'.\n"
        f"Fill in real values for each field based on the input.\n\n"
        f"Required output fields:\n{field_description}"
    )

    msgs: list = [
        {"role": "system", "content": full_system},
        {
            "role": "user",
            "content": (
                "Return ONLY a JSON object with actual data values — no markdown, no schema definitions.\n\n"
                f"INPUT:\n{user_content}"
            ),
        },
    ]

    def _call(client):
        messages = list(msgs)  # fresh copy per attempt
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content or ""
            content = _strip_markdown_fence(raw)

            # Detect if model returned schema instead of data
            try:
                parsed_check = json.loads(content)
                if isinstance(parsed_check, dict) and _is_schema_not_data(parsed_check):
                    # Tell the model it returned schema, ask for data
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": (
                            "You returned the schema definition, not actual data. "
                            "Please return a JSON object with REAL VALUES filled in "
                            "(strings, numbers, arrays with items). "
                            "Do NOT include '$defs', 'properties', or 'type' keys."
                        ),
                    })
                    continue
            except json.JSONDecodeError:
                pass

            try:
                return response_model.model_validate_json(content)
            except (ValidationError, json.JSONDecodeError) as e:
                last_error = e
                # Add error feedback for next attempt
                if attempt < max_retries:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": f"Invalid JSON or missing fields: {str(e)[:200]}. Try again with valid data.",
                    })

        raise last_error or RuntimeError("Failed to parse LLM JSON response")

    return _call_with_fallback(_call, _call)


def call_llm_chat(
    *,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float = 0.7,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    def _call(client):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    return _call_with_fallback(_call, _call)
