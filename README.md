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

Use `validate-count` to compare the word count of a source PDF with its converted Markdown file. This provides a quick way to identify potentially missing content after conversion.

```bash
uv run validate-count path/to/document.pdf path/to/document.md
```

Example:

```bash
uv run validate-count "C:\Users\user\Downloads\document.pdf" "output\document.md"
```

Example output:

```text
PDF words:      48,214
Markdown words: 45,640
Difference:     -2,574
Coverage:       94.66%
```

The output shows:

- **PDF words** — Number of words extracted from the PDF text layer.
- **Markdown words** — Number of words in the converted Markdown file.
- **Difference** — Difference between the Markdown and PDF word counts.
- **Coverage** — Markdown word count as a percentage of the PDF word count.

A lower Markdown word count does not necessarily mean content is missing. Differences may result from removed page numbers, headers, footers, OCR artifacts, repeated text, or normalization during conversion. The coverage value should therefore be used as a preliminary validation signal. Unusually low coverage may indicate that the converted Markdown requires manual inspection.

> **Note:** PDF word counting depends on the PDF having an extractable text or OCR layer. Image-only scanned PDFs may require OCR before a meaningful comparison can be made.

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
uv run validate-count PDF MARKDOWN
```

Arguments:

| Argument | Required | Description |
| --- | --- | --- |
| `PDF` | Yes | Source PDF file used for the conversion. |
| `MARKDOWN` | Yes | Converted Markdown file to compare against the source PDF. |

Example:

```bash
uv run validate-count path/to/document.pdf path/to/document.md
```