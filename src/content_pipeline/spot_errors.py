import argparse
import json
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

# ----------------------------------------------------------------------
MAX_RETRIES = 5
INITIAL_BACKOFF = 10
API_DELAY = 0.5          # seconds between validation calls
MAX_SEGMENT_CHARS = 8000 # not used heavily now, but kept for safety

# ----------------------------------------------------------------------
def to_word_set(text: str) -> set[str]:
    return set(re.findall(r'[^\W\d_]+', text.lower()))

def _retry_call(client, model, contents, config=None):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {"model": model, "contents": contents}
            if config:
                kwargs["config"] = config
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                raise
            backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
            print(f"\n   Retry {attempt}/{MAX_RETRIES} in {backoff}s ({exc})", file=sys.stderr)
            time.sleep(backoff)
    raise last_exc

# ----------------------------------------------------------------------
# 1. Extract PDF text (unchanged)
# ----------------------------------------------------------------------
def extract_page_texts(pdf_path: str, client: genai.Client, model: str) -> dict[int, str]:
    print(f"📤 Uploading {pdf_path} ...", end="", flush=True)
    uploaded = client.files.upload(file=pdf_path)
    print(" done.")
    print("📄 Extracting text page by page ...", file=sys.stderr)
    prompt = (
        "Extract all text from this PDF, preserving language and script. "
        "Output page by page like:\n"
        "Page 1:\n[text]\nPage 2:\n[text]\n..."
    )
    response = _retry_call(client, model, [uploaded, prompt])
    pages = {}
    for m in re.finditer(r"Page\s+(\d+):\s*(.*?)(?=Page\s+\d+:|$)", response.text, re.DOTALL):
        pages[int(m.group(1))] = m.group(2).strip()
    return pages

# ----------------------------------------------------------------------
# 2. Split Markdown into paragraphs and map to PDF pages
# ----------------------------------------------------------------------
def split_into_paragraphs(md_text: str) -> list[dict]:
    """Return a list of paragraphs with their start/end positions."""
    paras = re.split(r"\n\s*\n", md_text)
    result = []
    pos = 0
    for p in paras:
        start = md_text.find(p, pos)
        if start == -1:
            continue
        end = start + len(p)
        result.append({"text": p, "start": start, "end": end})
        pos = end
    return result

def sequential_align_paragraphs(paragraphs: list[dict], pdf_pages: dict[int, str]) -> list[int]:
    """
    For each paragraph, find the best matching page (sequential order).
    Returns a list of page numbers (one per paragraph). Pages may repeat.
    """
    sorted_pages = sorted(pdf_pages.keys())
    if not sorted_pages:
        return [0] * len(paragraphs)

    assignments = []
    page_idx = 0
    # Look ahead up to 4 pages to find best match for each paragraph
    search_window = 4
    for para in paragraphs:
        words = to_word_set(para["text"])
        if not words:
            assignments.append(sorted_pages[page_idx] if page_idx < len(sorted_pages) else sorted_pages[-1])
            continue
        best_page = None
        best_score = 0
        end_idx = min(page_idx + search_window, len(sorted_pages))
        for idx in range(page_idx, end_idx):
            p = sorted_pages[idx]
            page_text = pdf_pages.get(p, "")
            page_words = to_word_set(page_text)
            if not page_words:
                continue
            overlap = len(words & page_words) / len(words)
            if overlap > best_score:
                best_score = overlap
                best_page = p
        if best_page is None:
            # fallback: use previous page or first page
            best_page = sorted_pages[page_idx] if page_idx < len(sorted_pages) else sorted_pages[-1]
        assignments.append(best_page)
        # Advance page pointer to the page after the best match
        page_idx = sorted_pages.index(best_page) + 1
        if page_idx >= len(sorted_pages):
            page_idx = len(sorted_pages) - 1
    return assignments

# ----------------------------------------------------------------------
# 3. Suspicious pattern detection
# ----------------------------------------------------------------------
def find_suspicious(paragraph: dict, page_num: int) -> list[dict]:
    """
    Apply rules to a paragraph's text and return a list of suspicious segments.
    Each segment: {"start": abs_start, "end": abs_end, "reason": str, "page": page_num}
    where start/end are absolute positions in the full Markdown.
    """
    text = paragraph["text"]
    offset = paragraph["start"]
    suspicious = []

    # Rule a: mid-word line break inside the paragraph
    # We look for a line that ends with a letter and the next line starts with a letter
    # This often indicates a split word.
    # We'll scan the paragraph text line by line.
    lines = text.split('\n')
    for i in range(len(lines) - 1):
        line = lines[i].rstrip()
        next_line = lines[i+1].lstrip()
        if not line or not next_line:
            continue
        # Check if line ends with a Devanagari letter (range \u0900-\u097F)
        if re.search(r'[\u0900-\u097F]$', line) and re.search(r'^[\u0900-\u097F]', next_line):
            # This is a potential broken word. Mark the area from end of line to beginning of next.
            # We'll include the last word of the line and the first word of the next line.
            last_word = re.findall(r'[\u0900-\u097F]+', line)
            first_word = re.findall(r'[\u0900-\u097F]+', next_line)
            if last_word and first_word:
                # Find positions within the paragraph
                # We need to find the start of last_word in the line and the end of first_word in next_line.
                # Simpler: mark the entire line ending + next line start as a segment.
                # But we want a small segment for the footnote. We'll take the last 20 chars of line and first 20 of next_line.
                seg_start_in_para = 0
                # Build the absolute position by summing lengths of previous lines + newlines
                # We'll do a simpler approach: just search for the concatenation of line+'\n'+next_line in paragraph text.
                # However, we can compute the offset by counting characters up to this line.
                # Use the line index to find absolute start.
                # Let's find the absolute position of this line's end.
                current_pos = offset
                for j in range(i):
                    current_pos += len(lines[j]) + 1  # +1 for newline
                # current_pos now points to the start of line i.
                # The end of line i is current_pos + len(line)
                line_end_abs = current_pos + len(line)
                # The start of next line is line_end_abs + 1
                next_line_start_abs = line_end_abs + 1
                # We'll flag a small window around the break: from line_end_abs-20 to next_line_start_abs+20
                seg_start = max(offset, line_end_abs - 20)
                seg_end = min(paragraph["end"], next_line_start_abs + 20)
                suspicious.append({
                    "start": seg_start,
                    "end": seg_end,
                    "reason": "Possible mid‑word line break",
                    "page": page_num
                })
                # Only flag one per line pair to avoid duplicates
                break

    # Rule b: stray Latin characters inside Hindi words
    # We search for a pattern: (Hindi letter)+ Latin_letter (Hindi letter)*
    # or standalone Latin letters in a Hindi context.
    for m in re.finditer(r'[\u0900-\u097F]*[A-Za-z]+[\u0900-\u097F]*', text):
        # Exclude if the whole paragraph is Latin (unlikely)
        # Mark the Latin sequence and a bit of surrounding Hindi
        s = m.start()
        e = m.end()
        seg_start = max(offset, offset + s - 5)
        seg_end = min(paragraph["end"], offset + e + 5)
        suspicious.append({
            "start": seg_start,
            "end": seg_end,
            "reason": "Latin characters inside Hindi text",
            "page": page_num
        })

    # Rule c: repeated word (e.g., "बहुत बहुत")
    for m in re.finditer(r'\b(\w+)\s+\1\b', text):
        s = m.start()
        e = m.end()
        seg_start = max(offset, offset + s - 10)
        seg_end = min(paragraph["end"], offset + e + 10)
        suspicious.append({
            "start": seg_start,
            "end": seg_end,
            "reason": "Repeated word",
            "page": page_num
        })

    # Rule d: very short line (<10 chars) that ends without punctuation
    for line in lines:
        stripped = line.strip()
        if 0 < len(stripped) < 10 and not re.search(r'[।॥\-]$', stripped):
            # find its position
            line_start_in_para = 0
            # We can find by searching the line in the paragraph text
            idx = text.find(line)
            if idx != -1:
                seg_start = offset + idx
                seg_end = seg_start + len(line)
                suspicious.append({
                    "start": seg_start,
                    "end": seg_end,
                    "reason": "Short cut‑off line",
                    "page": page_num
                })

    return suspicious

# ----------------------------------------------------------------------
# 4. Validate suspicious segment with Gemini against original PDF
# ----------------------------------------------------------------------
def validate_segment(segment: dict, pdf_pages: dict[int, str], client: genai.Client, model: str) -> dict | None:
    """
    Send the segment and its surrounding original page text to Gemini.
    If it's an error, return a dict with "correction", "note", and the segment's start/end.
    If not an error, return None.
    """
    page_num = segment["page"]
    # Use a window of pages around the assigned page (e.g., ±1)
    pages_to_check = set([page_num, page_num - 1, page_num + 1])
    orig_parts = []
    for p in sorted(pages_to_check):
        if p in pdf_pages:
            orig_parts.append(f"Page {p}:\n{pdf_pages[p]}")
    original_text = "\n\n".join(orig_parts)

    # The exact suspicious text
    suspect_text = segment["text"]

    prompt = f"""You are an expert proofreader for Hindi/Devanagari OCR errors.
Below is a small excerpt from a Markdown file that was flagged as suspicious: "{segment['reason']}".
The original PDF text from the corresponding page(s) is also provided.

Examine the suspicious excerpt and compare it with the original.
If the excerpt contains a genuine OCR error (typo, wrong character, missing word, a word without context, semantic errors,  etc.), return a JSON object with:
- "start": the 0‑based index inside the excerpt where the error begins.
- "end": the exclusive end index.
- "correction": the correct text.
- "note": a brief explanation.

If the excerpt is actually correct and matches the original (the suspicious pattern was just a false alarm), return exactly:
{{"error": false}}

Output ONLY the JSON object, no other text.

Original PDF text:
{original_text}

Suspicious Markdown excerpt:
{suspect_text}
"""
    config = types.GenerateContentConfig(response_mime_type="application/json")
    response = _retry_call(client, model, [prompt], config=config)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("error") is False:
            return None   # false alarm
        if "start" in data and "end" in data:
            return data
    except Exception as e:
        print(f"\n⚠️  Validation JSON parse error: {e}", file=sys.stderr)
    return None

# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Smart OCR error spotter (validate only suspicious segments).")
    parser.add_argument("--md", "-m", required=True, nargs="+")
    parser.add_argument("--pdf", "-p", required=True, nargs="+")
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--block-pattern", default="markdown", help="Not used for detection, but kept for compatibility.")
    args = parser.parse_args()

    if len(args.md) != len(args.pdf):
        print("❌ --md and --pdf count mismatch", file=sys.stderr)
        sys.exit(1)

    client = genai.Client()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for md_path, pdf_path in zip(args.md, args.pdf):
        print(f"\n{'='*60}")
        print(f"📄 Document: {pdf_path}  ↔  {md_path}")
        print(f"{'='*60}")

        if not Path(md_path).exists() or not Path(pdf_path).exists():
            print("❌ File missing.", file=sys.stderr)
            continue

        # ----- Extract PDF -----
        try:
            pdf_pages = extract_page_texts(pdf_path, client, args.model)
        except Exception as e:
            print(f"❌ PDF extraction failed: {e}", file=sys.stderr)
            continue
        print(f"📖 PDF pages: {len(pdf_pages)}")

        # ----- Read Markdown and split into paragraphs -----
        flawed_md = Path(md_path).read_text(encoding="utf-8")
        paragraphs = split_into_paragraphs(flawed_md)
        print(f"📝 Paragraphs: {len(paragraphs)}")

        # ----- Map paragraphs to PDF pages -----
        page_assignments = sequential_align_paragraphs(paragraphs, pdf_pages)
        # Attach page number to each paragraph
        for i, para in enumerate(paragraphs):
            para["page"] = page_assignments[i]

        # ----- Scan for suspicious segments -----
        print("🔍 Scanning for suspicious patterns ...")
        all_suspicious = []
        for para in paragraphs:
            segs = find_suspicious(para, para["page"])
            for seg in segs:
                # Also store the actual text of the segment for later validation
                seg["text"] = flawed_md[seg["start"]:seg["end"]]
                all_suspicious.append(seg)

        print(f"   Found {len(all_suspicious)} potential issues.")
        if not all_suspicious:
            print("✅ No suspicious segments – output unchanged.")
            continue

        # ----- Validate each suspicious segment with Gemini (check against PDF) -----
        confirmed_errors = []  # (global_position, note)
        for i, seg in enumerate(all_suspicious):
            print(f"   Verifying {i+1}/{len(all_suspicious)}: {seg['reason']} ...", end="", flush=True)
            try:
                result = validate_segment(seg, pdf_pages, client, args.model)
            except Exception as e:
                print(f" failed ({e})")
                continue
            if result is None:
                print(" false alarm")
            else:
                # Global position = segment start + error start
                global_pos = seg["start"] + result["start"]
                note = result.get("note", "Corrected")
                confirmed_errors.append((global_pos, note))
                print(" confirmed ✅")
            time.sleep(API_DELAY)

        # ----- Insert footnotes -----
        confirmed_errors.sort(key=lambda x: x[0], reverse=True)
        annotated = list(flawed_md)
        footnotes = []
        for idx, (pos, note) in enumerate(confirmed_errors, 1):
            marker = f"[^{idx}]"
            annotated[pos:pos] = marker
            footnotes.append(f"[^{idx}]: {note}")

        result = "".join(annotated)
        if footnotes:
            result += "\n\n## Footnotes\n" + "\n".join(footnotes) + "\n"

        out_path = output_dir / (Path(md_path).stem + "_annotated.md")
        out_path.write_text(result, encoding="utf-8")
        print(f"\n✅ Completed: {len(footnotes)} real errors footnoted → {out_path}")

if __name__ == "__main__":
    main()