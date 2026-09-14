import re
import os
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
        "nhạc cụ": "Elective_Management"
    }
    detected_codes = []
    question_lower = question.lower()
    for name, code in subject_map.items():
        if name in question_lower:
            detected_codes.append(code)
    return detected_codes

def retrieve(question: str, top_k: int = 20):
    """Truy xuất tài liệu: Cho phép TÌM KIẾM ĐỒNG THỜI Môn học + Khung chương trình"""
    query_embedding = embedder.encode(question).tolist()
    
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

    is_curriculum_query = any(keyword in question.lower() for keyword in curriculum_keywords)
    
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

def build_prompt(question: str, results: dict):
    if not results["documents"] or not results["documents"][0]:
         return None, []

    context_blocks = []
    sources = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_blocks.append(f"[Nguồn: {meta.get('source_file')} - {meta.get('section_title')}]\n{doc}")
        sources.append(meta.get("source_file"))
        
    context = "\n\n".join(context_blocks)
    
    # Ép AI CỐ GẮNG SUY LUẬN thay vì bỏ cuộc sớm
    prompt = f"""Bạn là chuyên gia tư vấn học vụ xuất sắc của trường Đại học FPT chuyên ngành Software Engineering.
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
        
    return prompt, list(set(sources))

def call_llm(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Quota" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return "Hệ thống đang tạm thời quá tải do vượt mức 5 câu hỏi/phút. Bạn vui lòng đợi 30 giây rồi hỏi lại mình nhé!"
        return "Xin lỗi, đã có lỗi kết nối xảy ra với AI. Vui lòng thử lại sau."

if __name__ == "__main__":
    test_question = "Môn học nào ở kỳ 6 cho phép sinh viên đi làm thực tế tại doanh nghiệp?"
    docs = retrieve(test_question)
    prompt_text, source_files = build_prompt(test_question, docs)
    if prompt_text:
        answer = call_llm(prompt_text)
        print(f"\n[AI Trả lời]:\n{answer}")
        print(f"\n[Nguồn]: {source_files}")