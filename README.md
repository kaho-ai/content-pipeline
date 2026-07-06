# Content Pipeline

Small Python tools for content conversion workflows.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python, dependency, and command management.

Install uv if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, sync the environment:

```bash
uv sync
```

You can also skip the explicit sync step and run commands directly with `uv run`; uv will create `.venv` and install dependencies as needed.

## Credentials

The PDF converter uses Gemini through `google-genai`. Set `GEMINI_API_KEY` before running conversions:

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

Generated Markdown files are written as `<output-dir>/<pdf-name>.md`. The tool writes one combined Markdown file per input PDF; it does not currently write per-page or per-chunk Markdown files.

## CLI Reference

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
| `--model MODEL` | No | `gemini-3.1-flash-lite-preview` | Gemini model used for conversion. |
| `-h, --help` | No | None | Show command help and exit. |

Examples:

```bash
uv run gemini-md -o /tmp/markdown /home/dman/Downloads/tinyspec.pdf
uv run gemini-md --output-dir markdown --chunk-size 10 large-document.pdf
uv run gemini-md --output-dir markdown --model gemini-3.1-flash-lite-preview document.pdf
```
