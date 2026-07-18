import sys
import fitz
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("Usage: validate-count <pdf_file> <markdown_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    md_path = sys.argv[2]

    # PDF word count
    doc = fitz.open(pdf_path)
    pdf_text = "".join(page.get_text() for page in doc)
    pdf_words = len(pdf_text.split())

    # Markdown word count
    md_text = Path(md_path).read_text(encoding="utf-8")
    md_words = len(md_text.split())

    # Comparison
    difference = md_words - pdf_words
    coverage = (md_words / pdf_words * 100) if pdf_words else 0

    print(f"PDF words:      {pdf_words:,}")
    print(f"Markdown words: {md_words:,}")
    print(f"Difference:     {difference:+,}")
    print(f"Coverage:       {coverage:.2f}%")


if __name__ == "__main__":
    main()