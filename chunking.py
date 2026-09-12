import re

def chunk_by_heading(content: str, min_len: int = 30):
    """Tách nội dung markdown thành các chunk theo heading (## hoặc ###)"""
    # Tách văn bản mỗi khi gặp dấu xuống dòng kèm theo 1 đến 3 dấu #
    parts = re.split(r'\n(?=#{1,3}\s)', content)
    chunks = []
    
    for part in parts:
        text = part.strip()
        if len(text) >= min_len:
            # Tìm tiêu đề của đoạn để làm metadata
            title_match = re.match(r'#{1,3}\s+(.+)', text)
            section_title = title_match.group(1) if title_match else "Nội dung chính"
            chunks.append({"section_title": section_title, "text": text})
            
    return chunks