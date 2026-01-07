import re

# Mocking the current simplified implementation
MAX_SENTENCE_LENGTH = 80

def _normalize_text_current(text: str) -> str:
    if not text:
        return ""
    # Simulate current logic:
    # 1. Number conversion (skipped for this test as it's not the root cause of splitting)
    # 2. regex strip
    text = re.sub(r"[A-Za-z0-9]+", " ", text) # Simplified
    text = re.sub(r"[^\u4e00-\u9fff，。！？、；：,.!?…~\s]", " ", text)
    # 3. Collapse whitespace (THE ISSUE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _split_long_text(text: str) -> list:
    # Exact logic from current code
    sentence_delimiters = r'([。！？!?…~；;]|\n)'
    parts = re.split(sentence_delimiters, text)
    
    sentences = []
    current = ""
    for i, part in enumerate(parts):
        if not part: continue
        if re.match(sentence_delimiters, part):
            current += part
            if current.strip(): sentences.append(current.strip())
            current = ""
        else:
            current = part
    if current.strip(): sentences.append(current.strip())
    
    result = []
    buffer = ""
    
    for sentence in sentences:
        if len(sentence) > MAX_SENTENCE_LENGTH:
            if buffer:
                result.append(buffer)
                buffer = ""
            sub_sentences = re.split(r'([，,、])', sentence)
            sub_buffer = ""
            for sub in sub_sentences:
                if len(sub_buffer) + len(sub) <= MAX_SENTENCE_LENGTH:
                    sub_buffer += sub
                else:
                    if sub_buffer: result.append(sub_buffer)
                    sub_buffer = sub
            if sub_buffer: result.append(sub_buffer)
        elif len(buffer) + len(sentence) <= MAX_SENTENCE_LENGTH:
            buffer += sentence
        else:
            if buffer: result.append(buffer)
            buffer = sentence
    if buffer: result.append(buffer)
    return result

def _normalize_text_proposed(text: str) -> str:
    if not text: return ""
    
    # PROPOSED FIX: Convert newlines to punctuation BEFORE collapsing
    text = text.replace('\n', '。')
    
    text = re.sub(r"[A-Za-z0-9]+", " ", text)
    text = re.sub(r"[^\u4e00-\u9fff，。！？、；：,.!?…~\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

input_text = """为您准备了3个生活小贴士：

1、日常生活小窍门：定期整理衣柜，只保留最近一年穿过的衣服

2、节约用水技巧：洗澡时间控制在5-10分钟，使用节水淋浴头

3、环保生活习惯：随手关灯，使用可重复使用的购物袋"""

print("--- Current Behavior ---")
normalized = _normalize_text_current(input_text)
print(f"Normalized: {normalized}")
segments = _split_long_text(normalized)
print(f"Segments ({len(segments)}):")
for s in segments:
    print(f"[{len(s)}] {s}")

print("\n--- Proposed Behavior ---")
normalized_new = _normalize_text_proposed(input_text)
print(f"Normalized: {normalized_new}")
# Proposed logic allows `_split_long_text` to see the '。' we added
segments_new = _split_long_text(normalized_new)
print(f"Segments ({len(segments_new)}):")
for s in segments_new:
    print(f"[{len(s)}] {s}")
