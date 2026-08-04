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
uv run gemini-md --language Hindi --output-dir markdown path/to/document.pdf
```

This uses the asynchronous Gemini Batch API by default. The command submits all
pending page-batch conversions together, polls the job until it finishes, then
submits the three AI validation checks together in a second Batch API job. Only
page batches that fail conversion or validation are submitted again.

Use synchronous `generate_content` requests instead when immediate interactive
processing is preferable:

```bash
uv run gemini-md --sync --language Hindi --output-dir markdown path/to/document.pdf
```

[Gemini Batch API jobs](https://ai.google.dev/gemini-api/docs/batch-api) may take
substantially longer than synchronous requests; Google specifies a target
turnaround time of up to 24 hours. Pressing Ctrl-C while the command is polling
attempts to cancel the active remote batch job.

Convert multiple PDFs in one run:

```bash
uv run gemini-md --language Hindi --output-dir markdown first.pdf second.pdf third.pdf
```

Convert a specific page from a PDF:

```bash
uv run gemini-md --language Hindi --output-dir markdown --pages 3 path/to/document.pdf
```

Convert a range of pages from a PDF:

```bash
uv run gemini-md --language Hindi --output-dir markdown --pages 3-15 path/to/document.pdf
```

Generated Markdown files are written as `<output-dir>/<pdf-name>.md`. When `--pages` is used, only the selected page or page range is converted. The selected pages are combined into a single Markdown output file rather than being written as separate per-page files.

PDFs are processed as ordered batches of explicit source pages according to `--batch-size` (default: `10` pages). Every batch retains its original PDF page numbers, and the resulting batch Markdown is combined into one file per input PDF.

Gemini must emit an internal `<!-- source-page: N -->` marker for every page in a batch. Those boundaries let validation and progress output identify the exact original PDF page, and they are removed before the final Markdown file is written. Each batch is checked for complete ordered page markers and Unicode script consistency with the required `--language`, then by three focused Gemini validators covering completeness and ordering, transcription fidelity, and Markdown structure. Punctuation, numbers, symbols, and whitespace are allowed regardless of script; every letter and combining mark must use the declared language's script. Every ordinary foreign-script text occurrence is printed as a concise, non-failing `WARNING [FOREIGN_LANGUAGE_TEXT][page N] 'text'` review item. Foreign-script text inside a URL is printed as a non-failing `WARNING [FOREIGN_LANGUAGE_URL][page N] 'url'` review item. These warnings do not retry the batch; other validation failures retry it with combined feedback, using the same five-attempt retry limit as other conversion failures.

If Gemini returns `PROHIBITED_CONTENT` without text, the retry feedback states
that the input is public-domain literary text being converted faithfully for
archival purposes and was flagged incorrectly. The reported finish reason is
preserved in terminal output instead of being reduced to a generic empty-response
message. A blocked multi-page request is split into one request per source page;
all of those single-page requests are submitted together in the next Gemini
Batch API job, validated independently, and reassembled in source-page order.

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
uv run gemini-md --language LANGUAGE [OPTIONS] PDF [PDF ...]
```

Arguments:

| Argument | Required | Description |
| --- | --- | --- |
| `PDF` | Yes | One or more PDF files to convert. |

Options:

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `-o, --output-dir OUTPUT_DIR` | Yes | None | Directory where Markdown files will be written. The directory is created if it does not exist. |
| `--language LANGUAGE` | Yes | None | Language of the source book. The same language applies to every input PDF and determines the Unicode script allowed in converted text. Supported languages include Assamese, Awadhi, Bengali, Braj, English, Gujarati, Hindi, Kannada, Khari Boli, Malayalam, Marathi, Nepali, Odia, Persian, Punjabi, Sanskrit, Tamil, Telugu, and Urdu. |
| `--batch-size BATCH_SIZE` | No | `10` | Number of explicit source pages included in each Gemini batch. Must be greater than zero. |
| `--pages PAGES` | No | None | Convert a specific page or page range (for example, `3` or `3-15`) instead of the entire PDF. When multiple PDFs are provided, the same page selection is applied to each input. |
| `--model MODEL` | No | `gemini-3.5-flash-lite` | Gemini model used for conversion. |
| `--sync` | No | Disabled | Override the default asynchronous Gemini Batch API and use synchronous `generate_content` calls. |
| `-h, --help` | No | None | Show command help and exit. |

Examples:

```bash
uv run gemini-md --language Hindi -o /tmp/markdown /home/dman/Downloads/tinyspec.pdf
uv run gemini-md --language Hindi --output-dir markdown --batch-size 10 large-document.pdf
uv run gemini-md --language Hindi --output-dir markdown --pages 3-15 /home/dman/Downloads/tinyspec.pdf
uv run gemini-md --language Hindi --output-dir markdown --model gemini-3.5-flash-lite document.pdf
uv run gemini-md --sync --language Hindi --output-dir markdown document.pdf
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
