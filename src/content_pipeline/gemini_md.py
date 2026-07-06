from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import pypdf
from google import genai

CHUNK_SIZE = 20  # pages per chunk
MAX_RETRIES = 5
INITIAL_BACKOFF = 10  # seconds
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


def split_pdf(input_file: Path, chunk_size: int, tmp_dir: Path) -> list[Path]:
    """Split a PDF into chunks of chunk_size pages."""
    reader = pypdf.PdfReader(input_file)
    total_pages = len(reader.pages)
    chunk_paths: list[Path] = []

    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)
        writer = pypdf.PdfWriter()
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])

        chunk_path = tmp_dir / f"{input_file.stem}_chunk_{start + 1:04d}-{end:04d}.pdf"
        with chunk_path.open("wb") as f:
            writer.write(f)
        chunk_paths.append(chunk_path)

    print(f"  Split into {len(chunk_paths)} chunks of up to {chunk_size} pages ({total_pages} total pages)")
    return chunk_paths


def extract_pdf_page(input_file: Path, reader: pypdf.PdfReader, page_number: int, tmp_dir: Path) -> Path:
    """Write a single 1-based PDF page to a temporary PDF."""
    writer = pypdf.PdfWriter()
    writer.add_page(reader.pages[page_number - 1])

    page_path = tmp_dir / f"{input_file.stem}_page_{page_number:04d}.pdf"
    with page_path.open("wb") as f:
        writer.write(f)
    return page_path


def convert_chunk(
    client: genai.Client,
    chunk_path: Path,
    chunk_index: int,
    total_chunks: int,
    model: str,
) -> str:
    """Upload a chunk to Gemini, convert it to Markdown, and retry transient failures."""
    prompt = """
Analyze this document and convert it entirely into Markdown.
Preserve the structure perfectly, including headings, bullet points, and tables.
Output ONLY the raw markdown. Do not include introductory or concluding conversational text.
"""
    uploaded_file = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

            uploaded_file = client.files.upload(file=chunk_path)

            response = client.models.generate_content(
                model=model,
                contents=[uploaded_file, prompt],
            )

            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

            text = response.text
            if not text or not text.strip():
                raise ValueError("Empty response from Gemini")

            print(f"  Chunk {chunk_index + 1}/{total_chunks} converted successfully")
            return text

        except Exception as exc:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass
                uploaded_file = None

            if attempt == MAX_RETRIES:
                print(f"  Chunk {chunk_index + 1}/{total_chunks} FAILED after {MAX_RETRIES} attempts: {exc}")
                raise

            backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
            print(f"  Chunk {chunk_index + 1}/{total_chunks} attempt {attempt} failed: {exc}")
            print(f"    Retrying in {backoff}s...")
            time.sleep(backoff)

    raise RuntimeError("unreachable retry state")


def convert_pdf(
    input_file: Path,
    output_dir: Path,
    client: genai.Client,
    chunk_size: int = CHUNK_SIZE,
    model: str = DEFAULT_MODEL,
    page_number: int | None = None,
) -> Path:
    """Split a large PDF, convert each chunk, concatenate the results, and write Markdown."""
    if not input_file.exists():
        raise FileNotFoundError(f"File '{input_file}' not found.")
    if not input_file.is_file():
        raise ValueError(f"Path '{input_file}' is not a file.")

    print(f"\n{'=' * 60}")
    print(f"Processing: {input_file}")
    print(f"{'=' * 60}")

    reader = pypdf.PdfReader(input_file)
    total_pages = len(reader.pages)
    print(f"  Total pages: {total_pages}")

    if page_number is not None:
        if page_number > total_pages:
            raise ValueError(f"Page {page_number} is out of range for a {total_pages}-page PDF.")

        print(f"  Single-page mode - converting page {page_number}...")
        with tempfile.TemporaryDirectory() as tmp_name:
            page_path = extract_pdf_page(input_file, reader, page_number, Path(tmp_name))
            md_text = convert_chunk(client, page_path, 0, 1, model)
    elif total_pages <= chunk_size:
        print("  Small file - converting directly...")
        md_text = convert_chunk(client, input_file, 0, 1, model)
    else:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp_dir = Path(tmp_name)
            chunk_paths = split_pdf(input_file, chunk_size, tmp_dir)

            md_parts = []
            for i, chunk_path in enumerate(chunk_paths):
                try:
                    md = convert_chunk(client, chunk_path, i, len(chunk_paths), model)
                    md_parts.append(md)
                except Exception:
                    print(f"  Skipping chunk {i + 1} due to repeated failures.")
                    md_parts.append(
                        "\n\n"
                        f"<!-- CONVERSION FAILED: pages {i * chunk_size + 1}-"
                        f"{min((i + 1) * chunk_size, total_pages)} -->"
                        "\n\n"
                    )

            md_text = "\n\n".join(md_parts)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{input_file.stem}_page_{page_number}" if page_number is not None else input_file.stem
    output_file = output_dir / f"{output_stem}.md"
    output_file.write_text(md_text, encoding="utf-8")

    print(f"  Success! Markdown saved to '{output_file}'")
    return output_file


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF files to Markdown with Gemini.")
    parser.add_argument("pdf_files", nargs="+", type=Path, metavar="PDF", help="PDF file to convert.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where Markdown files will be written.",
    )
    parser.add_argument(
        "--chunk-size",
        type=positive_int,
        default=CHUNK_SIZE,
        help=f"Pages per Gemini request for large PDFs. Defaults to {CHUNK_SIZE}.",
    )
    parser.add_argument(
        "--page",
        type=positive_int,
        help="Convert only this 1-based page number instead of the entire PDF.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use. Defaults to {DEFAULT_MODEL}.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    client = genai.Client()
    failures = 0

    for pdf_file in args.pdf_files:
        try:
            convert_pdf(
                input_file=pdf_file,
                output_dir=args.output_dir,
                client=client,
                chunk_size=args.chunk_size,
                model=args.model,
                page_number=args.page,
            )
        except Exception as exc:
            failures += 1
            print(f"Error converting '{pdf_file}': {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
