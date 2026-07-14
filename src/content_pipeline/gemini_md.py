from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import pypdf
from google import genai

CHUNK_SIZE = 10  # pages per chunk
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


def extract_pdf_range(
    input_file: Path,
    reader: pypdf.PdfReader,
    start_page: int,
    end_page: int,
    tmp_dir: Path,
) -> Path:
    """Write a 1-based inclusive page range to a temporary PDF."""

    writer = pypdf.PdfWriter()

    for page in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page])

    range_path = (
        tmp_dir /
        f"{input_file.stem}_{start_page:04d}-{end_page:04d}.pdf"
    )

    with range_path.open("wb") as f:
        writer.write(f)

    return range_path

def parse_page_range(value: str) -> tuple[int, int]:
    """
    Accepts:
        5
        5-20
    """

    if "-" not in value:
        page = positive_int(value)
        return page, page

    try:
        start, end = value.split("-", 1)
        start = positive_int(start)
        end = positive_int(end)
    except Exception:
        raise argparse.ArgumentTypeError(
            "Expected PAGE or START-END (e.g. 5 or 5-20)"
        )

    if start > end:
        raise argparse.ArgumentTypeError(
            "Start page must be <= end page."
        )

    return start, end


def convert_chunk(
    client: genai.Client,
    chunk_path: Path,
    chunk_index: int,
    total_chunks: int,
    model: str,
) -> str:
    """Upload a chunk to Gemini, convert it to Markdown, and retry transient failures."""
    prompt = """
"You are a Pandit-level digital archivist and Chhandashastra (छंदशास्त्र) scholar specialising in Hindi, Braj, Awadhi, and archaic Khari Boli poetry/prose from OCR-scanned antique PDFs. Your task is to convert the scanned Devanagari text into pristine, structurally perfect Markdown. DO NOT summarize, modernize, or translate the poetry into English or modern Hindi.

**GENERAL PRINCIPLE** --- WHEN IN DOUBT, PRESERVE: Every rule below asks you to make a judgment call: is this an OCR error or an archaic spelling? Is this number a page reference or a verse reference? Does this line break because the metrical foot ends, or because the page margin cut it off? 
Whenever the evidence is genuinely ambiguous, resolve in favour of preserving the text exactly as scanned rather than "correcting" it. A wrongly preserved OCR glitch is a small, visible, fixable error; a wrongly "corrected" archaic form or a deleted verse number is a silent, undetectable loss of the original.

**1. Devanagari OCR Correction (Conservative & Contextual):** Scan errors are common. Correct clear OCR mistakes based on context, but DO NOT alter intentional archaic spellings. Pay special attention to commonly confused Devanagari characters: - श vs स vs ष (e.g., correcting "सिव" to "शिव" if contextually religious, but leaving "सो" as is). - त vs थ vs त्र. - ब vs व (very common OCR swap). - च vs छ. - Matra errors: ि (short i) vs ी (long i), ु vs ू. Restore them based on the metrical meter (Chhand) if ambiguous --- see Rule 4 for matra-count references. 
- Halant (क्) omissions or insertions---fix only if it breaks the word beyond recognition. Many conjuncts render without a visible halant in normal typesetting; don't insert one just because the component sounds are there. 
- संयुक्ताक्षर (conjunct characters) such as क्ष, ज्ञ, श्र, द्ध, ट्ट: OCR often splits these into their separate base consonants. Reassemble the conjunct wherever the split reading isn't a real independent word. 
- Avagraha (ऽ), marking vowel elision (e.g., सो ऽहं): easily dropped by scanners. Restore where its absence would be grammatically or metrically odd, but don't insert one speculatively. - झ vs भ: OCR sometimes confuses these when the vertical stroke is faded. Use word‑recognition to decide: e.g., if the word is likely "झूठ" (falsehood) but reads "भूठ", correct to झूठ; if it is "भक्त" (devotee) but reads "झक्त", correct to भक्त.
- If uncertain, preserve the scanned glyph and add a bracketed note like [संदिग्ध: झ/भ].

**2. **Historical Devanagari Glyph Recognition & Nasal Mark Restoration (Mandatory):**

Many books printed before the Government of India's 1960--62 Devanagari standardization use historical ("Calcutta-style") glyphs. These are purely typographic variants and NEVER represent different letters. Always recognize the underlying character, not merely the visual shape.

Historical glyphs that commonly appear include archaic/alternate printed forms of:

-   अ, झ, ण, ल, श, क्ष

These MUST always be transcribed as their modern Unicode equivalents. Never substitute them with another character because they resemble one.
In particular:

-   Never confuse archaic अ with त्र, प, फ, ट, य, or any conjunct.
-   Never confuse archaic झ with भ or ध.
-   Never confuse archaic ण with व, ब, द, ल, ग or any similar-looking glyph.
-   Always use contextual word recognition before visual similarity. If the surrounding word forms a valid Hindi, Braj, Awadhi, Sanskrit, or Prakrit word, prefer that interpretation over the raw OCR shape.
-   When uncertain, preserve the historically correct letter rather than inventing another character based solely on appearance.

------------------------------------------------------------------------

**Mandatory Restoration of Nasal Marks (ं, ँ, ः)**

- OCR frequently omits or corrupts Anusvāra (ं), Chandrabindu (ँ), and Visarga (ः). Before producing the final Markdown, perform a dedicated proofreading pass whose sole purpose is restoring missing nasal marks wherever grammar or established spelling requires them.

- Never leave these missing merely because the printed dot is faint or absent.

- Pay special attention to extremely common grammatical forms, including but not limited to:  
Locative and postpositional words: में, नहीं, कहीं, यहीं, वहीं etc. Never output: मे, नही, कही, यही, वही etc. ie, without Anusvāra (ं) unless the source unmistakably uses that spelling intentionally.

- Always restore plural oblique endings such as: लोगों, मित्रों, भक्तों, गुरुओं, बालकों, राजाओं, स्त्रियों, भाइयों, बहनों, कवियों, देवताओं, पुत्रों etc. Never drop the anusvāra or chandrabindu from these endings. Likewise, restore missing nasal marks in extremely common lexical words whenever the intended word is obvious from context, including words such as: संग, संगत, प्रसंग, संबंध, संपर्क, संभावना, संसार, संपूर्ण, संपत्ति, संस्कृति, संघर्ष, etc.

For example, OCR may incorrectly produce: सग, परसग, सबध, सपर्क, सभावना, ससार, etc. ie without Anusvāra (ं).  
These should be restored to their correct spellings whenever the surrounding context makes the intended word unambiguous.

------------------------------------------------------------------------

**Lexical Recognition Takes Priority**

If the OCR output forms a non-word or an impossible grammatical construction, but changing one or two visually similar characters (including restoring missing nasal marks) produces a valid and contextually appropriate Hindi, Braj, Awadhi, or Sanskrit word, make that correction. Do not preserve obvious OCR errors merely because they resemble the scanned glyph.

------------------------------------------------------------------------

**Mandatory Final Orthographic Verification**

Before returning the Markdown, perform one complete proofreading pass
and verify that:

1.  Every historical Devanagari glyph has been interpreted as its correct Unicode character.
2.  No archaic glyph has been mistaken for another letter based solely on visual similarity.
3.  Every missing Anusvāra (ं), Chandrabindu (ँ), and Visarga (ः) required by grammar has been restored.
4.  Common postpositions (especially "में"), plural oblique endings, and common Sanskrit/Hindi lexical words have not lost their nasal marks.
5.  The final text reads as grammatically correct Hindi/Braj/Awadhi/Sanskrit while preserving genuine historical spellings and never modernizing archaic language.

**3. Preserve Archaic Language (Absolutely Critical):** DO NOT modernize Braj Bhasha, Awadhi, or old Hindi to contemporary standard Hindi.
Preserve old verb forms (e.g., कहत not कहते, रह्यौ not रहे, ह्वै not होकर, तजि not त्याग कर). Preserve पद्य (verse) vocabulary exactly as scanned. A form that looks unusual but follows a recognisable Braj/Awadhi grammatical pattern (verb endings in -त, -यौ, -हिं, -ई, etc.) is intentional, not an OCR error.

**4. Punctuation: Devanagari vs. English:** **Strictly preserve** the Devanagari पूर्ण विराम (।) and the दोहरा दण्ड (॥). **DO NOT** replace them with English periods (.) or commas. Use ॥ to denote the absolute end of a verse/poem or a major section. Preserve Devanagari numerals (१, २, ३...) exactly as they appear in verse or couplet numbering rather than converting them to Arabic numerals.

**5. Distinguishing Line Wraps from True Metrical Breaks (Crucial for Chhand):** Hindi poetry relies on मात्रा (syllabic weight) and तुक (rhyme), NOT capitalization (which doesn't exist in Devanagari).
Reference matra counts for the common metres here, so foot-completion is a concrete check rather than a guess: 
- दोहा: 13 + 11 matras per line, two lines total, with तुक on the 11-matra halves. 
- सोरठा: the reverse pattern --- 11 + 13 matras per line. 
- चौपाई: 16 matras per चरण (quarter), four quarters, usually typeset as two lines of 32. If the source names a छंद other than these three and you aren't confident of its matra count, don't guess --- default to the line breaks as scanned rather than reconstructing an unfamiliar metre. 
**Rule**: If a line wraps to the next margin or across a page break, but the metrical foot is incomplete by the counts above, **merge** it with the previous line.
**Rule**: Break into a new line only when: - You encounter a clear metrical pause (end of a Doha/Chaupai foot per the counts above). - You see a ॥ at the end. - The rhyme scheme (तुकांत) clearly shifts to a new couplet. 
**Stanza breaks**: Always insert a blank line between distinct stanzas, dohas, or sorthas.

**6. Indentation and Alignment (Width-Safe):** Many Hindi poems use indentation for visual balance (especially मुक्तक or सवैया), and some verses are centered on the physical page. Preserve the RELATIVE indentation pattern --- which lines sit further right than others --- rather than copying the exact absolute space count from the scan.  The original page may be much wider than the screen this Markdown will actually be read on (many readers will view it on a phone), so a gap that looked centered or balanced on a wide printed page can shove a short line almost entirely to one side, or off the visible area, when reproduced as the same number of literal spaces in a narrow window.  Cap leading whitespace at roughly 12--16 spaces on any single line, using small, consistent steps (about 4 spaces per indent level) to show relative offset, even if the scan's own gap is wider than that. If a line appears as an isolated, heavily-indented fragment with no visible counterpart before or beside it, that is almost always a scan-rendering artifact rather than content to reproduce literally --- bring it back to a small, sensible indent instead of preserving an extreme one.

**7. Paratextual Elements: What to Keep, What to Remove (Strict):** 
**Delete** running headers (repeated book or chapter titles printed at the top of a page) and printing marks. 
**Delete** catchwords (the first word of the next page repeated at the bottom of the current one). 

**Keep page and poem numbers --- do not delete them.** -> Mark every page number using this exact decorative template, regardless of how plain or elaborate that specific page's own print styling looks: `<!-- page : 1 -->` --- replace 1 with the actual page number, in whichever numeral system it's actually printed in (Arabic or Devanagari; don't convert one to the other). Use this same template consistently for every page in the document. Do not simplify it to a bare number, a short dash pair, or any other shorter style, and do not invent a different decorative style even if the scanned page's own flourish looks different --- this is a normalized marker for the converted document, not a literal transcription of each page's individual print ornament. This includes page numbers that appear in editorial brackets in the source (e.g., [१०]): still normalize them to the template above. A page-number marker will often fall in the middle of a verse, simply because the physical page break happens to land inside a line. When that happens: extract the marker so it doesn't fracture the verse text, rejoin the verse's two halves per Rule 4 as if the break had never happened, and place the page-number marker on its own line at the nearest stanza boundary (immediately before or after, whichever is closer). 

**Don't confuse page numbers with verse numbers.**
A page number is a standalone marker disconnected from any specific verse, simply counting pages in sequence; a verse number (e.g., "॥ २ ॥") sits directly against the verse it closes, counting compositions, not pages. Keep both, but keep them visually distinct: page numbers use the template above on their own line at a stanza boundary; verse numbers stay attached to the verse they close, unchanged.

**8. Titles, Subheadings, and Invocations (मंगलाचरण):** - Main book title → # (H1). - Section/Ramayana-type chapters (काण्ड) or major divisions → ## (H2). - A named composition that opens a genuinely distinct section --- a titled hymn, a labelled sub-episode, or the first shift into a new grouping --- → ### (H3). Do NOT give every individual दोहा, सोरठा, or चौपाई its own H3 as it recurs through a continuous passage; a single kand-length section can contain hundreds of each, and heading every one defeats the purpose of a heading hierarchy. Where you still want to flag the type of a recurring verse without a full heading, use a small inline label instead, e.g. 
**(दोहा)**. - Dedications or invocations (e.g., "ॐ","श्री गणेशाय नमः") keep as standalone italic lines using *...*. 
  - Sanskrit framing verses: many Awadhi/Braj devotional works (Tulsidas's Ramcharitmanas is the best-known example) open each major division with several Sanskrit श्लोक --- invocations distinct from the vernacular narrative that follows --- and often close it with a short Sanskrit passage too. If the source does this, keep these shlokas in Sanskrit exactly as scanned, set apart with a label such as "(संस्कृत श्लोक)" or a blockquote, and do NOT apply Rule 2's Braj/Awadhi archaic-verb-form logic to them --- they follow Sanskrit grammar, not Awadhi conjugation, so "correcting" them toward Awadhi forms would be as wrong as modernizing them.

**9. Footnotes / टिप्पणी:** If there are footnotes (marked by *, †, or superscript numbers), convert them to Markdown footnotes ([^1]). Keep footnote identifiers unique across the entire document, not just within one poem --- e.g., number continuously, or prefix with a section tag like [^bk12-1] for a Bal Kand footnote. Most Markdown renderers treat footnote IDs as global to the whole document, so resetting to [^1] in every poem will make later definitions silently overwrite or fail to resolve earlier ones. Still place each footnote's definition at the end of its own poem, separated by a horizontal rule (---), so it stays visually close to its context even though the identifier itself is unique document-wide.

**10. Tables of Contents (अनुक्रमणिका):** If present, convert to a nested Markdown unordered list (- ), preserving indentation levels. Since page numbers are now preserved as markers throughout the body (Rule 6), keep the TOC's page-number references too --- they correspond to real, findable points in the converted document.

**11. Structural Containers for Pandoc/Typst (Custom Divs):** To enable later styling in pandoc/typst, wrap the content in the following fenced divs as appropriate. These divs do not alter the text; they are purely structural markers for output rendering.

-   **`::: poem`** -- Use for any metrical composition: dohas, chaupais, sorthas, songs, shlokas (whether vernacular or Sanskrit), ghazals, and all verse. Wrap a **continuous block of verse** (without prose interruption) in this div. If a major section (e.g., a काण्ड) consists entirely of verse, wrap the entire section in one `::: poem` div; if there are prose commentaries or interspersed passages, wrap each separate verse passage individually. The div should start at the beginning of the verse block (after any heading or invocation) and close at the end of that verse block, before a heading or prose section. Do **not** wrap individual lines or single dohas unless they stand alone as a self‑contained poem.

-   **`::: quote`** -- Use for long quotations or extracted passages that are **not** in verse---e.g., a prose excerpt from another text, a cited passage, or a block of dialogue that is clearly borrowed.

-   **`::: letter`** -- Use if the text is explicitly a letter (e.g., begins with a salutation and ends with a signature). Do not use for poetic epistles that are already in verse (they go under `::: poem`).

-   **`::: preface`** -- Use for the author's preface, introduction, foreword, or any front matter that is not part of the main narrative. This includes editorial introductions if they are part of the scanned source.

-   **`::: footnote`** -- Use for lengthy editorial notes or appendixed remarks that are not linked via superscript numbers in the main text. (Normal scholarly footnotes with markers should still be handled as per Rule 8.) This div is rarely needed; use it only when you encounter a distinct section labelled as "footnote" or "टिप्पणी" that stands apart from the verse.

-   **`::: appendix`** -- Use for any material clearly designated as an appendix, such as supplementary tables, glossaries, or extra chapters outside the main sequence.

**Placement and nesting**: These divs should enclose entire logical sections. Page‑number markers (Rule 6) must remain **outside** the divs, as they are navigational aids and not part of the original content. Do not nest a div inside another div of the same type; if two verse blocks are separated by prose, close the first `::: poem` before the prose and reopen after it. Always close each div with `:::` on its own line. The div syntax follows pandoc's fenced div format.

**12. Mandatory Page‑by‑Page Processing (No Skipping):** Process every page of the source strictly in order. Do **not** skip, condense, merge, or silently omit any page, even if it seems repetitive, damaged, or hard to read. Transcribe what is legible; for entirely illegible portions, insert a bracketed note such as `[अस्पष्ट]` or `[पृष्ठ X – पूर्णतः अस्पष्ट]` and continue to the next page. If a page is blank, output only the page‑number marker (Rule 6) and a line `[रिक्त पृष्ठ]`. At the end of your conversion, verify that the sequence of page markers is continuous and includes every page from the first to the last; if any marker is missing, re‑examine your output.

**Output Format:** Return your entire response inside a single, clean fenced Markdown code block. Open and close the wrapper with four backtick characters together --- one more than the usual three --- immediately followed by the word markdown on the opening line. This way, if the source document happens to contain its own triple-backtick block for any reason, that inner fence can't prematurely close your outer wrapper.

Markdown nohighlight, # This stays a code block but Tree-sitter highlighting is skipped.

Process every page of the source, strictly in order. Do not skip, condense, merge, or silently omit any page, even one that seems repetitive, damaged, or hard to read --- transcribe what is legible and mark an unclear portion with something like [अस्पष्ट] rather than dropping the page.

Do not include any explanatory text, summaries, romanization, or feedback outside this code block. Begin directly with the converted content, and contain nothing inside the block except the converted text itself --- no commentary on the conversion process, no sentence noting that a section or page range is complete, and no note or guess about where a future response should resume. If the document is too long to finish in a single response, simply stop at the end of a poem or kand --- a clean structural boundary, never mid-verse --- and close the fence there. The last page-number marker already present in your own output is the resumption point; do not restate or re-estimate it in prose, since that kind of self-generated guess is redundant at best and has already produced an incorrect page number once."
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
    page_range: tuple[int, int] | None = None, 
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

    working_file = input_file

    if page_range is not None:
        start_page, end_page = page_range

        if end_page > total_pages:
            raise ValueError(
                f"Page {end_page} exceeds PDF length ({total_pages})."
            )

        print(f"  Selecting pages {start_page}-{end_page}...")

        temp_dir = tempfile.TemporaryDirectory()

        working_file = extract_pdf_range(
            input_file,
            reader,
            start_page,
            end_page,
            Path(temp_dir.name),
        )

        reader = pypdf.PdfReader(working_file)
        total_pages = len(reader.pages)

    if total_pages <= chunk_size:
        print("  Small file - converting directly...")
        md_text = convert_chunk(client, working_file, 0, 1, model)
    else:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp_dir = Path(tmp_name)

            chunk_paths = split_pdf(
                working_file,
                chunk_size,
                tmp_dir,
            )

            md_parts = []

            for i, chunk_path in enumerate(chunk_paths):

                if i > 0:
                    print("  Sleeping for 3 seconds between requests...")
                    time.sleep(3)

                try:
                    md = convert_chunk(
                        client,
                        chunk_path,
                        i,
                        len(chunk_paths),
                        model,
                    )

                    md_parts.append(md)

                except Exception:
                     print(
                          f"  Skipping chunk {i + 1} due to repeated failures."
                         )

                     start = i * chunk_size + 1
                     end = min((i + 1) * chunk_size, total_pages)

                     md_parts.append(
                          f"\n\n<!-- CONVERSION FAILED: pages {start}-{end} -->\n\n"
                            )

            md_text = "\n\n".join(md_parts)

    output_dir.mkdir(parents=True, exist_ok=True)

    if page_range is None:
        output_stem = input_file.stem
    else:
        start, end = page_range

        if start == end:
            output_stem = f"{input_file.stem}_page_{start}"
        else:
            output_stem = f"{input_file.stem}_pages_{start}-{end}"
    
    output_file = output_dir / f"{output_stem}.md"
    output_file.write_text(md_text, encoding="utf-8")

    if temp_dir is not None:
        temp_dir.cleanup()

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
        "--pages",
        type=parse_page_range,
        metavar="PAGE|START-END",
        help="Convert only the specified page or page range.",
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
                page_range=args.pages,
            )
        except Exception as exc:
            failures += 1
            print(f"Error converting '{pdf_file}': {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
