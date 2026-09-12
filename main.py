from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_service import retrieve, build_prompt, call_llm

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Second Brain RAG API")

# Bật CORS để Flutter App có thể gọi API mà không bị trình duyệt/hệ thống chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định nghĩa cấu trúc dữ liệu Nhận vào (Request)
class ChatRequest(BaseModel):
    question: str

# Định nghĩa cấu trúc dữ liệu Trả về (Response)
class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

# Tạo Endpoint POST /chat
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    print(f"Nhận câu hỏi từ Flutter: {req.question}")
    
    # 1. Tìm kiếm tài liệu trong ChromaDB
    docs = retrieve(req.question)
    
    # 2. Xây dựng Prompt
    prompt, sources = build_prompt(req.question, docs)
    
    # 3. Gọi Gemini và trả kết quả
    if prompt:
        answer = call_llm(prompt)
        
        # NÂNG CẤP: Nếu AI từ chối, làm rỗng danh sách nguồn
        if "Xin lỗi, mình không tìm thấy" in answer:
            return ChatResponse(answer=answer, sources=[])
            
        return ChatResponse(answer=answer, sources=sources)
    else:
        return ChatResponse(
            answer="Xin lỗi, mình không tìm thấy thông tin liên quan trong tài liệu của trường.", 
            sources=[]
        )