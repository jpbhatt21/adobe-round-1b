# pdf_parser.py

import fitz, re

def max_overlap(a, b):
    max_olap = 0
    min_len = min(len(a), len(b))
    for i in range(1, min_len+1):
        if a[-i:] == b[:i]:
            max_olap = i
    return max_olap

def merge_fragments(fragments):
    result, i = [], 0
    while i < len(fragments):
        cur = fragments[i].copy()
        text = cur["text"]
        while i+1 < len(fragments):
            nxt = fragments[i+1]["text"]
            ol = max_overlap(text, nxt)
            if ol>0:
                text += nxt[ol:]
                i += 1
            else:
                break
        cur["text"] = text
        result.append(cur)
        i += 1
    return result

def extract_blocks_with_fonts(page):
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"]!=0: continue
        for line in b["lines"]:
            for span in line["spans"]:
                t = span["text"].strip()
                if not t: continue
                blocks.append({
                    "text": t,
                    "size": span["size"],
                    "font": span["font"],
                    "bbox": span["bbox"],
                    "page": page.number+1,
                    "is_bold": "Bold" in span["font"]
                })
    return blocks

def assign_header_levels(blocks):
    sizes = sorted({b["size"] for b in blocks}, reverse=True)
    size2lvl = {s: f"H{i+1}" for i,s in enumerate(sizes)}
    hdrs = []
    for b in blocks:
        words = b["text"].split()
        if len(words)>18: continue
        if b["is_bold"] or b["size"]>=sizes[0]*0.88 or len(words)<=10:
            hdrs.append({
                "level": size2lvl[b["size"]],
                "text":  b["text"],
                "page":  b["page"]
            })
    return hdrs

def dedup_and_clean_headers(headers):
    seen, out = set(), []
    for h in headers:
        norm = re.sub(r'\W','', h["text"]).lower()
        if norm in seen or len(h["text"])<3: continue
        seen.add(norm)
        out.append(h)
    return out

def find_best_title(headers, blocks):
    p1 = [b for b in blocks if b["page"]==1]
    if not p1: return "Untitled"
    max_sz = max(b["size"] for b in p1)
    cands = [b["text"] for b in p1 if abs(b["size"]-max_sz)<0.1]
    return " ".join(cands).strip() or "Untitled"

def extract_from_pdf_with_merge(pdf_path):
    doc = fitz.open(pdf_path)
    # 1) collect spans
    all_blocks = []
    for page in doc:
        all_blocks += extract_blocks_with_fonts(page)
    # 2) merge
    merged = merge_fragments(all_blocks)
    # 3) headers
    hdrs = assign_header_levels(merged)
    hdrs = dedup_and_clean_headers(hdrs)
    title = find_best_title(hdrs, merged)
    # 4) slice into sections
    secs = []
    for i,h in enumerate(hdrs):
        start = next(idx for idx,b in enumerate(merged)
                     if b["text"]==h["text"] and b["page"]==h["page"])
        end = len(merged)
        if i+1 < len(hdrs):
            nh = hdrs[i+1]
            end = next(idx for idx,b in enumerate(merged)
                       if b["text"]==nh["text"] and b["page"]==nh["page"])
        content = " ".join(b["text"] for b in merged[start+1:end])
        secs.append({
            "title":   h["text"],
            "level":   h["level"],
            "page":    h["page"],
            "content": content
        })
    return title, secs
