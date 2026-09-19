import re
import os
import json
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Tải biến môi trường (Lấy API Key từ file .env)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Đang kiểm tra các Model khả dụng với API Key của bạn...")
target_model_name = "gemini-3.6-flash"
print(f"✅ Đã chọn đích danh Model: {target_model_name}")

# Khởi tạo model
model = genai.GenerativeModel(target_model_name)

print("Đang khởi tạo RAG Service...")
# 2. Khởi tạo lại Model nhúng và kết nối DB
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="chroma_store")
collection = client.get_or_create_collection("second_brain")

def detect_subject_code(question: str):
    """Dùng Regex bắt TẤT CẢ mã môn và chuẩn hóa Hoa/Thường (VD: ITE302c, DPL303m)"""
    # Tìm tất cả các cụm từ khớp với định dạng mã môn (không phân biệt hoa thường)
    matches = re.findall(r'\b[a-zA-Z]{2,5}\d{2,3}[a-zA-Z]?\b', question)
    formatted_codes = []
    
    for match in matches:
        # Tách mã thành 3 phần: Tiền tố chữ - Khối số - Hậu tố chữ
        parts = re.match(r'([a-zA-Z]+)(\d+)([a-zA-Z]?)', match)
        if parts:
            prefix = parts.group(1).upper()   # Ví dụ: ITE
            digits = parts.group(2)           # Ví dụ: 302
            suffix = parts.group(3).lower()   # Ví dụ: c
            formatted_codes.append(f"{prefix}{digits}{suffix}")
            
    return formatted_codes

def extract_subjects_from_intent(question: str) -> list:
    """TỪ ĐIỂN NÂNG CẤP: Nhận diện ý định phức tạp của sinh viên"""
    subject_map = {
        "lịch sử đảng": "VNR202",
        "tư tưởng hcm": "HCM202",
        "tư tưởng hồ chí minh": "HCM202",
        "triết học": "MLN111",
        "kinh tế chính trị": "MLN122",
        "chủ nghĩa xã hội": "MLN131",
        "đồ án": "SEP490",
        "khởi nghiệp": "EXE401",
        "thực tập": "OJT202",
        "đi làm thực tế": "OJT202",
        "doanh nghiệp": "OJT202",
        "ojt": "OJT202",
        "game": "Combo_Management",
        "trò chơi": "Combo_Management",
        "ai": "Combo_Management",
        "trí tuệ nhân tạo": "Combo_Management",
        "y tế": "Combo_Management",
        "nhạc cụ": "Elective_Management",
        # Câu hỏi về danh sách / chi tiết combo chuyên ngành hẹp
        "combo": "Combo_Management",
        "chuyên ngành hẹp": "Combo_Management",
        "se_com": "Combo_Management",
        "phe_com": "Combo_Management",
        "java chuyên sâu": "Combo_Management",
        "lập trình .net": "Combo_Management",
        "react": "Combo_Management",
        "nodejs": "Combo_Management",
        "devsecops": "Combo_Management",
        "devops": "Combo_Management",
        "vi mạch": "Combo_Management",
        "kỹ sư cầu nối": "Combo_Management",
        "khdl": "Combo_Management",
        "khoa học dữ liệu": "Combo_Management",
    }
    detected_codes = []
    question_lower = question.lower()
    for name, code in subject_map.items():
        if name in question_lower:
            detected_codes.append(code)
    return detected_codes


def is_combo_catalog_query(question: str) -> bool:
    """Câu hỏi về danh sách / môn của combo chuyên ngành hẹp → ưu tiên Combo Management.md."""
    q = question.lower()
    keywords = [
        "combo",
        "chuyên ngành hẹp",
        "se_com",
        "phe_com",
        "danh sách combo",
        "các combo",
        "bao nhiêu combo",
        "combo nào",
        "định hướng",
        "chuyên sâu java",
        "java chuyên sâu",
        ".net programming",
        "lập trình .net",
        "liệt kê các môn",
        "liệt kê môn",
        "môn học của từng",
        "môn của từng combo",
    ]
    return any(k in q for k in keywords)


def wants_combo_course_listing(question: str) -> bool:
    """User muốn liệt kê mã môn từng combo (không chỉ tên combo)."""
    q = question.lower()
    signals = [
        "liệt kê",
        "các môn",
        "môn học",
        "môn của",
        "chi tiết môn",
        "mã môn",
        "từng combo",
        "từng chuyên ngành",
        "syllabus combo",
    ]
    return is_combo_catalog_query(question) and any(s in q for s in signals)


def _fetch_all_combo_management_chunks():
    """Lấy TOÀN BỘ chunk Combo Management.md (không phụ thuộc top_k vector)."""
    try:
        raw = collection.get(
            where={"source_id": "Combo_Management"},
            include=["documents", "metadatas"],
        )
    except Exception:
        return None
    if not raw or not raw.get("ids"):
        try:
            raw = collection.get(
                where={"type": "combo_management"},
                include=["documents", "metadatas"],
            )
        except Exception:
            return None
    if not raw or not raw.get("ids"):
        return None
    # Chroma get returns flat lists
    return {
        "ids": [raw["ids"]],
        "documents": [raw["documents"]],
        "metadatas": [raw["metadatas"]],
        "distances": [[0.0] * len(raw["ids"])],
    }


def normalize_subject_code(code: str) -> str:
    """Chuẩn hóa mã môn (ITE302c → ITE302c)."""
    parts = re.match(r'([a-zA-Z]+)(\d+)([a-zA-Z]?)', code.strip())
    if not parts:
        return code.strip()
    return f"{parts.group(1).upper()}{parts.group(2)}{parts.group(3).lower()}"


def _legacy_retrieve(question: str, top_k: int = 20):
    """Luồng retrieve gốc — không đổi hành vi khi không có context ép."""
    query_embedding = embedder.encode(question).tolist()

    # Ưu tiên tuyệt đối: câu hỏi combo → lấy HẾT Combo Management.md
    if is_combo_catalog_query(question):
        full = _fetch_all_combo_management_chunks()
        if full and full["documents"] and full["documents"][0]:
            # Bổ sung thêm subject_slot gần nhất nếu còn chỗ
            extra = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(10, top_k),
                where={"type": "subject_slot"},
            )
            if extra["documents"] and extra["documents"][0]:
                seen = set(full["ids"][0])
                for i, cid in enumerate(extra["ids"][0]):
                    if cid in seen:
                        continue
                    full["ids"][0].append(cid)
                    full["documents"][0].append(extra["documents"][0][i])
                    full["metadatas"][0].append(extra["metadatas"][0][i])
                    full["distances"][0].append(
                        (extra.get("distances") or [[0]])[0][i]
                        if extra.get("distances")
                        else 0.0
                    )
            return full
        where_combo = {
            "$or": [
                {"source_id": "Combo_Management"},
                {"type": "combo_management"},
                {"type": "subject_slot"},
            ]
        }
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k, 40),
            where=where_combo,
        )
        if results["documents"] and results["documents"][0]:
            return results
    
    # 1. Gom tất cả các mã tìm được từ Regex và Từ điển
    codes = []
    regex_codes = detect_subject_code(question)
    if regex_codes: 
        codes.extend(regex_codes) # Thêm tất cả mã regex bắt được
    codes.extend(extract_subjects_from_intent(question))
    
    # Xóa các mã trùng lặp
    codes = list(set(codes))
    
    curriculum_keywords = [
        "ngành", "chuyên ngành", "chương trình", "kỹ thuật phần mềm", 
        "tín chỉ", "kỳ", "học kỳ", "môn học nào", "tiên quyết", "điều kiện", 
        "trước đó", "học qua", "đầu vào", "khác nhau"
    ]

    # "chuyên ngành hẹp" / combo đã xử lý ở trên — tránh chỉ lấy curriculum chung
    is_curriculum_query = any(keyword in question.lower() for keyword in curriculum_keywords)
    if is_combo_catalog_query(question):
        is_curriculum_query = False
    
    # Logic Lọc $or: Lấy CẢ tài liệu môn học LẪN tài liệu chương trình
    where_conditions = []
    if codes:
        if len(codes) == 1:
            where_conditions.append({"source_id": codes[0]})
        else:
            where_conditions.append({"source_id": {"$in": codes}})
            
    if is_curriculum_query:
        where_conditions.append({"type": "curriculum"})
        
    where_clause = None
    if len(where_conditions) == 1:
        where_clause = where_conditions[0]
    elif len(where_conditions) > 1:
        where_clause = {"$or": where_conditions}

    if where_clause:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )
        if not results["documents"] or not results["documents"][0]:
            results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    else:
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
         
    return results


def _query_with_fallback(query_embedding, top_k, where_clause):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause,
    )
    if not results["documents"] or not results["documents"][0]:
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results


def retrieve(
    question: str,
    top_k: int = 20,
    force_source_ids=None,
    mode: str = "default",
    study_plan_input=None,
    source_file=None,
):
    """Truy xuất tài liệu. Không có context ép → giữ nguyên luồng legacy."""
    force_source_ids = force_source_ids or []
    force_source_ids = [normalize_subject_code(c) for c in force_source_ids if c]
    mode = (mode or "default").strip().lower()

    # Early exit: luồng cũ y hệt
    if mode == "default" and not force_source_ids and not study_plan_input:
        return _legacy_retrieve(question, top_k=top_k)

    query_text = question
    if study_plan_input:
        goal = ""
        if isinstance(study_plan_input, dict):
            goal = study_plan_input.get("goal") or ""
        query_text = f"{question} {goal} khung chương trình kỳ học combo AI"

    query_embedding = embedder.encode(query_text).tolist()
    effective_top_k = top_k
    if mode == "study_plan":
        effective_top_k = max(top_k, 30)

    where_conditions = []

    if force_source_ids:
        if len(force_source_ids) == 1:
            where_conditions.append({"source_id": force_source_ids[0]})
        else:
            where_conditions.append({"source_id": {"$in": force_source_ids}})

    if source_file and not force_source_ids:
        where_conditions.append({"source_file": source_file})

    if mode == "study_plan":
        where_conditions.append({"type": "curriculum"})
        where_conditions.append({"type": "subject_slot"})
        # Ưu tiên tài liệu quản lý combo/elective nếu có trong KB
        where_conditions.append({"source_id": "Combo_Management"})

    where_clause = None
    if len(where_conditions) == 1:
        where_clause = where_conditions[0]
    elif len(where_conditions) > 1:
        where_clause = {"$or": where_conditions}

    if where_clause:
        return _query_with_fallback(query_embedding, effective_top_k, where_clause)

    return collection.query(query_embeddings=[query_embedding], n_results=effective_top_k)


def _default_prompt(question: str, context: str) -> str:
    """Prompt mặc định — giữ nguyên nội dung cũ."""
    return f"""Bạn là chuyên gia tư vấn học vụ xuất sắc của trường Đại học FPT chuyên ngành Software Engineering.
Nhiệm vụ của bạn là giải đáp thắc mắc của sinh viên một cách thông minh, tận tâm và chính xác.

YÊU CẦU BẮT BUỘC:
1. NGUỒN DỮ LIỆU: PHẢI dựa trên "THÔNG TIN THAM KHẢO". Bạn được phép dịch tự do tài liệu Tiếng Anh sang Tiếng Việt.
2. CỐ GẮNG SUY LUẬN & ĐỐI CHIẾU: 
   - Với câu hỏi tìm môn học (VD: môn nào đi làm thực tế), hãy đọc nội dung các môn để nhận diện (như OJT202 - On the job training).
   - Với câu hỏi so sánh (VD: SEP490 và EXE401), hãy tổng hợp và liệt kê rõ ràng điểm giống/khác nhau về môn tiên quyết, tín chỉ.
   - Nếu tài liệu chỉ đề cập mã môn (VD: MLN111), hãy tự hiểu và giải thích nó là môn gì dựa trên ngữ cảnh, không được nói là không biết.
3. CHỐNG ẢO GIÁC LINH HOẠT: Không được bịa đặt ngoài tài liệu. Tuy nhiên, nếu tài liệu có manh mối (dù là mã môn hay tên tiếng Anh), HÃY CỐ GẮNG SUY LUẬN để trả lời. Chỉ nói "Xin lỗi, không tìm thấy" khi hoàn toàn không có bất kỳ manh mối nào.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI CỦA SINH VIÊN: {question}
TRẢ LỜI:"""


def _combo_catalog_prompt(question: str, context: str, catalog: str) -> str:
    """Ép AI liệt kê ĐỦ mã môn từng combo — không được chỉ nêu tên combo."""
    return f"""Bạn là chuyên gia tư vấn học vụ FPTU ngành Software Engineering (BIT_SE_K19B).

Sinh viên hỏi về combo chuyên ngành hẹp. Bạn PHẢI trả lời đủ 2 phần:

### PHẦN 1 — Số lượng
- Nêu rõ có bao nhiêu combo tổng / bao nhiêu SE / bao nhiêu PHE (theo CATALOG).

### PHẦN 2 — Liệt kê môn học TỪNG combo (BẮT BUỘC)
Với **MỖI** combo trong CATALOG, phải liệt kê đầy đủ:
- Tên combo
- Từng môn: **Kỳ** + **mã môn** + **tên môn** (vd: Kỳ 5: `HSF302` — Working with Spring Framework)

CẤM:
- Chỉ mô tả chung chung ("tập trung vào Java…") mà không ghi mã môn.
- Bỏ sót combo.
- Bảo sinh viên tự xem FLM thay vì liệt kê (chỉ được nhắc FLM ở cuối, sau khi đã liệt kê đủ).

CATALOG COMBO (nguồn chính — PHẢI dùng hết):
{catalog}

THÔNG TIN THAM KHẢO BỔ SUNG:
{context}

CÂU HỎI: {question}
TRẢ LỜI (Markdown, tiếng Việt):"""


def build_prompt(
    question: str,
    results: dict,
    mode: str = "default",
    action: str = None,
    excerpt: str = None,
    study_plan_input=None,
    force_source_ids=None,
):
    if not results["documents"] or not results["documents"][0]:
         return None, []

    context_blocks = []
    sources = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_blocks.append(f"[Nguồn: {meta.get('source_file')} - {meta.get('section_title')}]\n{doc}")
        sources.append(meta.get("source_file"))
        
    context = "\n\n".join(context_blocks)
    mode = (mode or "default").strip().lower()
    sources_unique = list(set(sources))

    # Luồng cũ: đúng template prompt hiện tại — trừ câu hỏi catalog combo
    if mode == "default" and not excerpt and not study_plan_input:
        if is_combo_catalog_query(question):
            try:
                from combo_resolver import build_full_combo_catalog_text
                catalog = build_full_combo_catalog_text()
            except Exception:
                catalog = context
            if "Combo Management.md" not in sources_unique:
                sources_unique.append("Combo Management.md")
            return _combo_catalog_prompt(question, context, catalog), sources_unique
        return _default_prompt(question, context), sources_unique

    if mode == "subject_focus":
        ids = ", ".join(force_source_ids or []) or "môn đang mở"
        prompt = f"""Bạn là chuyên gia tư vấn học vụ FPTU SE, đang tập trung 100% vào môn: {ids}.

YÊU CẦU:
1. CHỈ trả lời dựa trên THÔNG TIN THAM KHẢO của môn này. Không kéo nội dung môn khác.
2. Nếu sinh viên hỏi tóm tắt / ôn tập / flashcard: trả lời bằng Markdown rõ ràng (tóm tắt hoặc danh sách Q&A dạng flashcard).
3. Dịch tự do Anh→Việt khi cần. Không bịa ngoài tài liệu.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI CỦA SINH VIÊN: {question}
TRẢ LỜI:"""
        return prompt, sources_unique

    if mode == "quick_action":
        action_map = {
            "explain": "Giải thích rõ ràng, dễ hiểu như giáo sư AI cho sinh viên SE.",
            "translate_vi": "Dịch sang tiếng Việt tự nhiên, giữ thuật ngữ kỹ thuật quan trọng.",
            "summarize": "Tóm tắt ngắn gọn các ý chính.",
        }
        instruction = action_map.get(action or "explain", action_map["explain"])
        excerpt_block = excerpt or question
        prompt = f"""Bạn là Giáo sư AI hỗ trợ đọc đề cương môn học FPTU.

NHIỆM VỤ: {instruction}
Đoạn văn bản được chọn:
\"\"\"
{excerpt_block}
\"\"\"

Dùng THÔNG TIN THAM KHẢO (nếu có) để bổ sung ngữ cảnh môn học. Trả lời bằng tiếng Việt.

THÔNG TIN THAM KHẢO:
{context}

TRẢ LỜI:"""
        return prompt, sources_unique

    if mode == "study_plan":
        semester = 3
        goal = "AI Engineer"
        combo_track = None
        if isinstance(study_plan_input, dict):
            semester = int(study_plan_input.get("current_semester") or 3)
            goal = study_plan_input.get("goal") or goal
            combo_track = study_plan_input.get("combo_track")
        try:
            from combo_resolver import build_combo_hint_for_prompt
            combo_hint = build_combo_hint_for_prompt(goal, combo_track)
        except Exception:
            combo_hint = (
                "BẮT BUỘC: Không được để code SE_COM* / PHE_COM*. "
                "Phải ghi mã môn cụ thể (VD Java: HSF302, SBA301, MSS301; .NET: PRN212, PRN222, PRN232)."
            )
        prompt = f"""Bạn là cố vấn học tập FPTU ngành Software Engineering (BIT_SE_K19B).

Sinh viên đang ở Kỳ {semester}, mục tiêu nghề nghiệp: {goal}.
Dựa trên THÔNG TIN THAM KHẢO (khung chương trình, combo, môn học), hãy lập lộ trình từ Kỳ {semester} đến Kỳ 9.

{combo_hint}

QUY TẮC BẮT BUỘC VỀ COMBO / GDTC:
1. Trong khung chương trình có slot PHE_COM*1/2/3 và SE_COM*1/2/3/4_ELE — đây chỉ là chỗ trống.
2. Trong JSON trả về, field "code" PHẢI là mã môn thật (HSF302, PRN212, VOV124, …), KHÔNG BAO GIỜ để SE_COM*, PHE_COM*, SE_COM10.2_4, SE_COM＊1.
3. type="combo" cho môn chuyên ngành hẹp; type="elective" cho GDTC (VOV/COV); type="core" cho môn bắt buộc.
4. Ghi rõ trong "note": slot gốc → mã môn (vd. "SE_COM*1 · Java → HSF302").

TRẢ LỜI BẰNG JSON THUẦN (không markdown, không giải thích ngoài JSON), đúng schema:
{{
  "current_semester": {semester},
  "goal": "{goal}",
  "semesters": [
    {{
      "semester": {semester},
      "label": "Kỳ {semester}",
      "courses": [
        {{
          "code": "HSF302",
          "name": "Working with Spring Framework",
          "credits": 3,
          "type": "combo",
          "note": "SE_COM*1 · Java chuyên sâu → HSF302"
        }}
      ]
    }}
  ]
}}

type chỉ nhận: "core" | "elective" | "combo".
Ưu tiên môn liên quan mục tiêu nghề nghiệp. Chỉ dùng mã môn có trong tài liệu tham khảo khi có thể.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {question}
JSON:"""
        return prompt, sources_unique

    # Fallback an toàn → prompt mặc định
    return _default_prompt(question, context), sources_unique


def call_llm(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Quota" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return "Hệ thống đang tạm thời quá tải do vượt mức 5 câu hỏi/phút. Bạn vui lòng đợi 30 giây rồi hỏi lại mình nhé!"
        return "Xin lỗi, đã có lỗi kết nối xảy ra với AI. Vui lòng thử lại sau."


def extract_json_object(text: str):
    """Trích JSON object từ phản hồi LLM (có thể bọc trong ```json)."""
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
    """Validate tối thiểu schema study_plan; trả về dict sạch hoặc None."""
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


if __name__ == "__main__":
    test_question = "Môn học nào ở kỳ 6 cho phép sinh viên đi làm thực tế tại doanh nghiệp?"
    docs = retrieve(test_question)
    prompt_text, source_files = build_prompt(test_question, docs)
    if prompt_text:
        answer = call_llm(prompt_text)
        print(f"\n[AI Trả lời]:\n{answer}")
        print(f"\n[Nguồn]: {source_files}")
