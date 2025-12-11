import json


def parse_llm_json(raw: str):
    """
    Try to parse raw LLM output as JSON.
    If there's extra text, try to grab the first {...} block.
    Works for both single object and list of objects.
    """
    raw = raw.strip()
    # Direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Try to slice from first { to last }
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        json_str = raw[start:end]
        return json.loads(json_str)
    except Exception:
        raise ValueError("Model output is not valid JSON:\n" + raw)
