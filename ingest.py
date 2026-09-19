import os
import glob
import yaml
import chromadb
from sentence_transformers import SentenceTransformer
from chunking import chunk_by_heading

KB_DIR = "data/knowledge_base"

# Khởi tạo model nhúng (chạy offline, không tốn phí API)
print("Đang tải model Embedding...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Khởi tạo DB Vector lưu cục bộ
client = chromadb.PersistentClient(path="chroma_store")
collection = client.get_or_create_collection("second_brain")

def parse_frontmatter(raw_text):
    """Bóc tách siêu dữ liệu (YAML) nằm giữa 2 đường --- ở đầu file .md"""
    if raw_text.startswith("---"):
        parts = raw_text.split("---", 2)
        if len(parts) >= 3:
            fm = parts[1]
            body = parts[2]
            meta = yaml.safe_load(fm) or {}
            return meta, body.strip()
    return {}, raw_text

def ingest_all():
    files = sorted(glob.glob(os.path.join(KB_DIR, "**", "*.md"), recursive=True))
    print(f"Tìm thấy {len(files)} file Markdown. Bắt đầu nạp dữ liệu...")
    
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
            
        meta, body = parse_frontmatter(raw)
        chunks = chunk_by_heading(body)
        rel_path = os.path.relpath(path, KB_DIR).replace("\\", "/")
        
        for i, chunk in enumerate(chunks):
            # Dùng đường dẫn tương đối để 114 file (kể cả README trùng tên) không ghi đè nhau
            chunk_id = f"{rel_path}::{i}"
            
            # Chuyển text thành vector
            embedding = embedder.encode(chunk["text"]).tolist()
            
            # Dùng upsert: Nếu đã có thì cập nhật, chưa có thì thêm mới
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[{
                    "source_id": meta.get("id", ""),
                    "type": meta.get("type", "unknown"),
                    "tags": ",".join(meta.get("tags", [])),
                    "section_title": chunk["section_title"],
                    "source_file": rel_path,
                }]
            )
    print(f"✅ Đã nạp thành công {len(files)} file vào Vector Database!")

if __name__ == "__main__":
    ingest_all()