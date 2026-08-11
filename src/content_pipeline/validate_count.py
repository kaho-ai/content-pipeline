import sys
import re
from pathlib import Path
import fitz  # PyMuPDF

# Regex that matches one or more Unicode letters (any script) – a "word"
WORD_PATTERN = re.compile(r'[^\W\d_]+', re.UNICODE)

def count_words(text: str) -> int:
    """Count words in text using a Unicode‑aware word pattern."""
    return len(WORD_PATTERN.findall(text))

def main():
    if len(sys.argv) < 3:
        print("Usage: validate-count <pdf_file> <md_file> [--threshold PERCENT]")
        sys.exit(2)

    pdf_path = sys.argv[1]
    md_path = sys.argv[2]
    threshold = 95.0  # default coverage threshold (percent)

    # Parse optional threshold
    for arg in sys.argv[3:]:
        if arg.startswith("--threshold="):
            try:
                threshold = float(arg.split("=")[1])
            except ValueError:
                print("Error: threshold must be a number")
                sys.exit(2)

    # ---- PDF word count ----
    doc = fitz.open(pdf_path)
    pdf_text = "".join(page.get_text() for page in doc)
    doc.close()

    pdf_words = count_words(pdf_text)
    if pdf_words == 0:
        print("Warning: PDF appears to have no extractable text layer (scanned image?).")
        print("Consider running OCR on the PDF before validation.")
        # Still proceed with Markdown count to show partial info
        md_text = Path(md_path).read_text(encoding="utf-8")
        md_words = count_words(md_text)
        print(f"Markdown words: {md_words:,}")
        print("Cannot compute coverage (PDF words = 0).")
        sys.exit(1)  # Indicate a problem

    # ---- Markdown word count ----
    md_text = Path(md_path).read_text(encoding="utf-8")
    md_words = count_words(md_text)

    # ---- Comparison ----
    missing_words = pdf_words - md_words
    coverage = (md_words / pdf_words) * 100 if pdf_words > 0 else 0

    # Determine status based on coverage percentage
    if coverage >= threshold:
        status = "PASS"
    else:
        status = "REVIEW"

    print(f"PDF words:       {pdf_words:,}")
    print(f"Markdown words:  {md_words:,}")
    print(f"Missing words:   {missing_words:,}")
    print(f"Coverage:        {coverage:.2f}%")
    print(f"Threshold:       {threshold}%")
    print(f"Status:          {status}")

    if status == "REVIEW":
        print(f"\nAction: Markdown coverage below {threshold}%. Manual review recommended.")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()