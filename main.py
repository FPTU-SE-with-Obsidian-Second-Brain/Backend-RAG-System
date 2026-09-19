from typing import Optional, Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag_service import (
    retrieve,
    build_prompt,
    call_llm,
    extract_json_object,
    validate_study_plan,
)
from combo_resolver import expand_study_plan_combos

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Second Brain RAG API")

# Bật CORS để Flutter App có thể gọi API mà không bị trình duyệt/hệ thống chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudyPlanInput(BaseModel):
    current_semester: int = Field(default=3, ge=0, le=9)
    goal: str = "AI Engineer"
    combo_track: Optional[str] = None  # VD: "Java chuyên sâu", "lập trình .NET"


class StudyPlanCourse(BaseModel):
    code: str
    name: str
    credits: Optional[float] = None
    type: str = "core"
    note: str = ""


class StudyPlanSemester(BaseModel):
    semester: int
    label: str
    courses: list[StudyPlanCourse] = []


class StudyPlan(BaseModel):
    current_semester: int
    goal: str
    semesters: list[StudyPlanSemester]
    combo_track: Optional[str] = None
    pe_track: Optional[str] = None


# Định nghĩa cấu trúc dữ liệu Nhận vào (Request) — field mới đều optional
class ChatRequest(BaseModel):
    question: str
    source_ids: Optional[list[str]] = None
    source_file: Optional[str] = None
    mode: Optional[
        Literal["default", "subject_focus", "quick_action", "study_plan"]
    ] = "default"
    action: Optional[Literal["explain", "translate_vi", "summarize"]] = None
    excerpt: Optional[str] = None
    study_plan_input: Optional[StudyPlanInput] = None


# Định nghĩa cấu trúc dữ liệu Trả về (Response)
class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    study_plan: Optional[StudyPlan] = None


def _has_forced_context(req: ChatRequest) -> bool:
    return bool(req.source_ids) or bool(req.study_plan_input) or (
        req.mode and req.mode != "default"
    )


def _finalize_study_plan(validated: dict, study_input: dict | None) -> tuple:
    """Resolve SE_COM*/PHE_COM* → concrete codes; build user-facing answer."""
    goal = (study_input or {}).get("goal") or validated.get("goal") or ""
    track = (study_input or {}).get("combo_track")
    expanded = expand_study_plan_combos(
        validated, goal=goal, combo_track=track
    )
    warnings = expanded.pop("warnings", None) or []
    clean = {
        "current_semester": expanded.get("current_semester"),
        "goal": expanded.get("goal") or goal,
        "semesters": expanded.get("semesters") or [],
        "combo_track": expanded.get("combo_track"),
        "pe_track": expanded.get("pe_track"),
    }
    track_label = clean.get("combo_track") or "chuyên ngành hẹp"
    answer = (
        f"Đã lập lộ trình từ Kỳ {clean['current_semester']} → Kỳ 9, "
        f"mục tiêu: {clean['goal']} (combo: {track_label}). "
        f"Các slot SE_COM*/PHE_COM* đã được đổi thành mã môn cụ thể."
    )
    if warnings:
        answer += " Lưu ý: " + " ".join(warnings[:3])
    return clean, answer


# Tạo Endpoint POST /chat
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    mode = req.mode or "default"
    print(f"Nhận câu hỏi từ Flutter: {req.question} | mode={mode}")

    force_ids = req.source_ids or []
    study_input = None
    if req.study_plan_input:
        study_input = req.study_plan_input.model_dump()

    # 1. Tìm kiếm tài liệu trong ChromaDB
    if not _has_forced_context(req):
        docs = retrieve(req.question)
        prompt, sources = build_prompt(req.question, docs)
    else:
        docs = retrieve(
            req.question,
            force_source_ids=force_ids,
            mode=mode,
            study_plan_input=study_input,
            source_file=req.source_file,
        )
        prompt, sources = build_prompt(
            req.question,
            docs,
            mode=mode,
            action=req.action,
            excerpt=req.excerpt,
            study_plan_input=study_input,
            force_source_ids=force_ids,
        )

    # 2–3. Gọi Gemini và trả kết quả
    if not prompt:
        return ChatResponse(
            answer="Xin lỗi, mình không tìm thấy thông tin liên quan trong tài liệu của trường.",
            sources=[],
            study_plan=None,
        )

    answer = call_llm(prompt)

    if "Xin lỗi, mình không tìm thấy" in answer:
        return ChatResponse(answer=answer, sources=[], study_plan=None)

    study_plan = None
    if mode == "study_plan":
        parsed = extract_json_object(answer)
        validated = validate_study_plan(parsed) if parsed else None
        if not validated:
            retry_prompt = (
                prompt
                + "\n\nPHẢN HỒI TRƯỚC KHÔNG ĐÚNG JSON. Hãy trả lại CHỈ một JSON object hợp lệ, không markdown."
                + "\nNhắc lại: code phải là mã môn thật (HSF302, PRN212, VOV124…), không SE_COM*/PHE_COM*."
            )
            retry_answer = call_llm(retry_prompt)
            parsed = extract_json_object(retry_answer)
            validated = validate_study_plan(parsed) if parsed else None
            if validated:
                clean, answer = _finalize_study_plan(validated, study_input)
                study_plan = StudyPlan(**clean)
            else:
                answer = retry_answer if retry_answer else answer
        else:
            clean, answer = _finalize_study_plan(validated, study_input)
            study_plan = StudyPlan(**clean)

    return ChatResponse(answer=answer, sources=sources, study_plan=study_plan)
