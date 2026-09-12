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

# Google yêu cầu sử dụng đích danh bản 3.6-flash cho tài khoản của bạn
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
    """Dùng Regex bắt mã môn học (VD: PRM393, SWP391) để tìm kiếm chính xác hơn"""
    match = re.search(r'\b[A-Z]{2,5}\d{2,3}[a-zA-Z]?\b', question.upper())
    return match.group(0) if match else None

def retrieve(question: str, top_k: int = 5):
    """Truy xuất tài liệu từ Vector DB"""
    query_embedding = embedder.encode(question).tolist()
    code = detect_subject_code(question)
    
    # Chiến lược 1: Ưu tiên lọc chính xác theo mã môn trước
    where_filter = {"source_id": code} if code else None
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )
    
    # Chiến lược 2: Nếu lọc theo mã môn không ra kết quả, tìm kiếm ngữ nghĩa toàn cục
    if code and (not results["documents"] or not results["documents"][0]):
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
    return results

def build_prompt(question: str, results: dict):
    """Ghép tài liệu vào Prompt và ra lệnh cấm AI bịa đặt"""
    if not results["documents"] or not results["documents"][0]:
         return None, []

    context_blocks = []
    sources = []
    
    # Lắp ghép các đoạn text tìm được
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_blocks.append(f"[Nguồn: {meta.get('source_file')} - {meta.get('section_title')}]\n{doc}")
        sources.append(meta.get("source_file"))
        
    context = "\n\n".join(context_blocks)
    
    # Prompt siêu gắt gao - Ràng buộc AI
    # Prompt siêu gắt gao - Đã bổ sung Quyền Dịch Thuật
    prompt = f"""Bạn là trợ lý học vụ thông minh của trường Đại học FPT chuyên ngành Software Engineering.
Nhiệm vụ của bạn là giải đáp thắc mắc của sinh viên.

YÊU CẦU BẮT BUỘC:
1. CHỈ được trả lời dựa trên "THÔNG TIN THAM KHẢO" bên dưới. 
2. LƯU Ý QUAN TRỌNG: Thông tin tham khảo có thể bằng Tiếng Anh, bạn ĐƯỢC PHÉP dịch sang Tiếng Việt để trả lời sinh viên một cách tự nhiên.
3. Nếu không tìm thấy thông tin liên quan để trả lời, HÃY TỪ CHỐI bằng câu: "Xin lỗi, mình không tìm thấy thông tin này trong tài liệu của trường", TUYỆT ĐỐI KHÔNG ĐƯỢC BỊA ĐẶT.
4. Trả lời ngắn gọn, rõ ràng, súc tích bằng tiếng Việt.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI CỦA SINH VIÊN: {question}
TRẢ LỜI:"""
    
    # Trả về prompt và danh sách nguồn (đã xóa trùng lặp bằng set)
    return prompt, list(set(sources))

def call_llm(prompt: str) -> str:
    """Gửi Prompt lên Gemini để sinh câu trả lời"""
    response = model.generate_content(prompt)
    return response.text

# ==========================================
# KHU VỰC TEST TRỰC TIẾP TRONG TERMINAL
# ==========================================
if __name__ == "__main__":
    # Bạn có thể đổi câu hỏi ở đây để test thử các kịch bản khác nhau
    test_question = "Môn PRM393 học về những kiến thức gì?"
    
    print(f"\n[Người dùng hỏi]: {test_question}")
    print("Đang tìm kiếm tài liệu và suy nghĩ...")
    
    docs = retrieve(test_question)
    prompt_text, source_files = build_prompt(test_question, docs)
    
    if prompt_text:
        answer = call_llm(prompt_text)
        print(f"\n[AI Trả lời]:\n{answer}")
        print(f"\n[Nguồn trích xuất]: {source_files}")
    else:
        print("\n[AI Trả lời]: Xin lỗi, mình không tìm thấy thông tin trong DB.")