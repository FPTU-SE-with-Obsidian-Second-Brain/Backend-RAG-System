"""Smoke tests for study_plan JSON helpers (no Chroma/Gemini required)."""
import json
import sys
from pathlib import Path

# Allow importing rag_service helpers without loading heavy models:
# We only test pure functions by importing from a lightweight stub path.
ROOT = Path(__file__).resolve().parent

# Inline copies of pure helpers to avoid loading SentenceTransformer at import.
def extract_json_object(text: str):
    import re
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def validate_study_plan(data):
    if not isinstance(data, dict):
        return None
    if "semesters" not in data or not isinstance(data["semesters"], list):
        return None
    semesters = []
    for sem in data["semesters"]:
        if not isinstance(sem, dict):
            continue
        courses_out = []
        for c in sem.get("courses") or []:
            if not isinstance(c, dict) or not c.get("code"):
                continue
            courses_out.append(
                {
                    "code": str(c.get("code")),
                    "name": str(c.get("name") or c.get("code")),
                    "credits": c.get("credits") if isinstance(c.get("credits"), (int, float)) else None,
                    "type": str(c.get("type") or "core"),
                    "note": str(c.get("note") or ""),
                }
            )
        try:
            semester_num = int(sem.get("semester"))
        except (TypeError, ValueError):
            continue
        semesters.append(
            {
                "semester": semester_num,
                "label": str(sem.get("label") or f"Kỳ {semester_num}"),
                "courses": courses_out,
            }
        )
    if not semesters:
        return None
    try:
        current = int(data.get("current_semester") or semesters[0]["semester"])
    except (TypeError, ValueError):
        current = semesters[0]["semester"]
    return {
        "current_semester": current,
        "goal": str(data.get("goal") or ""),
        "semesters": semesters,
    }


def test_extract_fenced_json():
    text = '```json\n{"current_semester": 3, "goal": "AI", "semesters": [{"semester": 3, "label": "Kỳ 3", "courses": [{"code": "CSD201", "name": "DS"}]}]}\n```'
    data = extract_json_object(text)
    assert data is not None
    plan = validate_study_plan(data)
    assert plan is not None
    assert plan["current_semester"] == 3
    assert plan["semesters"][0]["courses"][0]["code"] == "CSD201"


def test_invalid_plan_returns_none():
    assert validate_study_plan({"foo": 1}) is None
    assert validate_study_plan(None) is None


if __name__ == "__main__":
    test_extract_fenced_json()
    test_invalid_plan_returns_none()
    print("OK: study_plan helper smoke tests passed")
    sys.exit(0)
