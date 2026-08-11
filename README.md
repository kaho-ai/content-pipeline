# Content Pipeline

Small Python tools for content conversion workflows.

## Setup

This project uses `uv` for Python, dependency, and command management.

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, sync the environment:

```bash
uv sync
```

You can also skip the explicit sync step and run commands directly with `uv run`; `uv` will create `.venv` and install dependencies as needed.

## Credentials

The PDF converter **uses** Gemini through `google-genai`. Set `GEMINI_API_KEY` before running conversions:

```bash
export GEMINI_API_KEY="..."
```

## PDF to Markdown

Convert a PDF to Markdown:

```bash
uv run gemini-md --output-dir markdown path/to/document.pdf
```

Convert multiple PDFs in one run:

```bash
uv run gemini-md --output-dir markdown first.pdf second.pdf third.pdf
```

Convert a specific page from a PDF:

```bash
uv run gemini-md --output-dir markdown --pages 3 path/to/document.pdf
```

Convert a range of pages from a PDF:

```bash
uv run gemini-md --output-dir markdown --pages 3-15 path/to/document.pdf
```

Generated Markdown files are written as `<output-dir>/<pdf-name>.md`. When `--pages` is used, only the selected page or page range is converted. The selected pages are combined into a single Markdown output file rather than being written as separate per-page files.

For full-document conversion, large PDFs are processed internally in chunks according to `--chunk-size` (default: `20` pages). The resulting chunks are combined into one Markdown file per input PDF.

## Validate PDF to Markdown

Use `validate-count` to compare the word count of a source PDF with its converted Markdown file. The tool provides a quick way to identify potentially missing content after conversion.

```bash
uv run validate-count path/to/document.pdf path/to/document.md
```

Optionally set a coverage threshold (default: 95%). If the Markdown’s word count falls below this percentage of the PDF’s word count, the status will be **REVIEW**.

```bash
uv run validate-count path/to/document.pdf path/to/document.md --threshold=90
```

Example:

```bash
uv run validate-count "C:\Users\user\Downloads\document.pdf" "output\document.md"
```

Example output:

```text
PDF words:       48,214
Markdown words:  45,640
Missing words:   2,574
Coverage:        94.66%
Threshold:       95%
Status:          REVIEW
```

The output shows:

- **PDF words** – Words extracted from the PDF text layer, counted using a Unicode‑aware word splitter (works for Hindi, English, and mixed scripts).
- **Markdown words** – Words in the converted Markdown file, counted identically.
- **Missing words** – PDF words minus Markdown words.
- **Coverage** – Markdown word count as a percentage of the PDF word count.
- **Threshold** – User‑configured minimum coverage (default 95%).
- **Status** – `PASS` if coverage ≥ threshold; `REVIEW` otherwise.

A lower Markdown word count does **not** necessarily mean content is missing. Differences may result from removed page numbers, headers, footers, OCR artifacts, or normalisation during conversion. The coverage percentage gives a preliminary signal – when it drops below your threshold, manual inspection is recommended.

> **Note:** PDF word counting depends on the PDF having an extractable text or OCR layer. If the PDF is an image‑only scan, `validate-count` will warn that no text layer is present and exit. Run OCR on the file first before using this validation.

## Spot OCR/Conversion Errors (Non‑destructive)

Use `spot-errors` to insert footnotes at every discrepancy between the original PDF and the converted Markdown, **without modifying the text**. This helps you manually review and correct only what’s actually wrong.

```bash
uv run spot-errors --md flawed.md --pdf original.pdf --output-dir annotated
```

It works with multiple pairs:

```bash
uv run spot-errors --md a.md b.md --pdf a.pdf b.pdf -o annotated
```

The output is a new Markdown file with added `[^1]`, `[^2]`, … markers and a `## Footnotes` section explaining each error.


## CLI Reference

### `gemini-md`

```bash
uv run gemini-md [OPTIONS] PDF [PDF ...]
```

Arguments:

| Argument | Required | Description |
| --- | --- | --- |
| `PDF` | Yes | One or more PDF files to convert. |

Options:

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `-o, --output-dir OUTPUT_DIR` | Yes | None | Directory where Markdown files will be written. The directory is created if it does not exist. |
| `--chunk-size CHUNK_SIZE` | No | `20` | Pages per Gemini request for large PDFs. Must be greater than zero. PDFs with at most this many pages are converted in a single request. |
| `--pages PAGES` | No | None | Convert a specific page or page range (for example, `3` or `3-15`) instead of the entire PDF. When multiple PDFs are provided, the same page selection is applied to each input. |
| `--model MODEL` | No | `gemini-3.1-flash-lite-preview` | Gemini model used for conversion. |
| `-h, --help` | No | None | Show command help and exit. |

Examples:

```bash
uv run gemini-md -o /tmp/markdown /home/dman/Downloads/tinyspec.pdf
uv run gemini-md --output-dir markdown --chunk-size 10 large-document.pdf
uv run gemini-md --output-dir markdown --pages 3-15 /home/dman/Downloads/tinyspec.pdf
uv run gemini-md --output-dir markdown --model gemini-3.1-flash-lite-preview document.pdf
```

### `validate-count`

```bash
uv run validate-count PDF MARKDOWN [--threshold PERCENT]
```

Arguments:

| Argument | Required | Description |
| --- | --- | --- |
| `PDF` | Yes | Source PDF file used for conversion. |
| `MARKDOWN` | Yes | Converted Markdown file to compare against the source PDF. |

Options:

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--threshold PERCENT` | No | `95` | Minimum coverage percentage for PASS status. If Markdown word count falls below this percentage of PDF words, the status is REVIEW. |
| `-h, --help` | No | None | Show command help and exit. |

Examples:

```bash
uv run validate-count document.pdf document.md
uv run validate-count document.pdf document.md --threshold=90
```

### `spot-errors`

```bash
uv run spot-errors --md FLAWED_MD... --pdf ORIGINAL_PDF... --output-dir DIR
```

Options:

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `-m, --md FLAWED_MD` | Yes | None | One or more flawed Markdown files (output of `gemini-md`). Repeat for multiple files. |
| `-p, --pdf ORIGINAL_PDF` | Yes | None | Original PDF files, in the same order and count as `--md`. |
| `-o, --output-dir DIR` | Yes | None | Directory where annotated Markdown files will be written. |
| `--api-key API_KEY` | No | `$GEMINI_API_KEY` | Gemini API key. If not given, the environment variable is used. |
| `-h, --help` | No | None | Show command help and exit. |

Examples:

```bash
uv run spot-errors --md flawed.md --pdf original.pdf --output-dir reviewed
uv run spot-errors -m ch1.md ch2.md -p ch1.pdf ch2.pdf -o reviewed
```

