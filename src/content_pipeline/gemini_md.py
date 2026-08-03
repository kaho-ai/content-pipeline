from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pypdf
from google import genai
from google.genai import types

BATCH_SIZE = 10  # pages per Gemini request
MAX_RETRIES = 5
INITIAL_BACKOFF = 10  # seconds
BATCH_POLL_INTERVAL = 30  # seconds
DEFAULT_MODEL = "gemini-3.5-flash-lite"
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>()\[\]{}]+",
    re.IGNORECASE,
)
PAGE_MARKER_PATTERN = re.compile(
    r"(?m)^<!-- source-page: ([1-9]\d*) -->[ \t]*$"
)

COMPLETENESS_CHECK = (
    "Completeness and ordering",
    """
Check that every source page is represented in order and that no legible
passage, stanza, heading, table entry, footnote, or other substantive content
was skipped, duplicated, condensed, or invented. Ignore running headers,
catchwords, printing marks, and standalone printed page numbers.
""",
)
TRANSCRIPTION_FIDELITY_CHECK = (
    "Transcription and orthographic fidelity",
    """
Check the candidate against the source for incorrect, missing, or invented
words and punctuation. Pay particular attention to Devanagari glyphs, matras,
conjuncts, Anusvara, Chandrabindu, Visarga, danda punctuation, verse numbers,
and obvious OCR confusions. Confirm that genuine Braj, Awadhi, Sanskrit, and
historical spellings were preserved rather than modernized, summarized, or
translated. Confirm that the text remains in the declared book language and
flag foreign-language substitutions even when they use the same Unicode script.
""",
)
MARKDOWN_OUTPUT_CHECK = (
    "Markdown structure and output contract",
    """
Check that the output is raw Markdown without surrounding code fences,
explanatory commentary, summaries, or completion notes. Confirm that headings,
lists, footnotes, stanza breaks, emphasis, and relative indentation preserve
the source structure and follow the conversion requirements.
""",
)
AI_VALIDATION_CHECKS = (
    COMPLETENESS_CHECK,
    TRANSCRIPTION_FIDELITY_CHECK,
    MARKDOWN_OUTPUT_CHECK,
)

PROHIBITED_CONTENT_RETRY_CONTEXT = (
    "This is a public-domain literary text being converted faithfully for "
    "archival purposes, and it has been flagged incorrectly. Continue the "
    "transcription without omitting the affected pages."
)


@dataclass(frozen=True)
class BatchPage:
    """One original PDF page included in a conversion batch."""

    number: int


@dataclass(frozen=True)
class PageBatch:
    """A Gemini request made from explicit, numbered source pages."""

    pdf_path: Path
    pages: tuple[BatchPage, ...]

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(page.number for page in self.pages)

    @property
    def page_label(self) -> str:
        first_page = self.pages[0].number
        last_page = self.pages[-1].number
        if first_page == last_page:
            return f"page {first_page}"
        return f"pages {first_page}-{last_page}"


@dataclass(frozen=True)
class ConversionUnit:
    """One independent request belonging to an original page batch."""

    parent_index: int
    batch: PageBatch


LANGUAGE_SCRIPTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "assamese": ("Assamese", ("BENGALI",)),
    "awadhi": ("Awadhi", ("DEVANAGARI", "VEDIC")),
    "bengali": ("Bengali", ("BENGALI",)),
    "braj": ("Braj", ("DEVANAGARI", "VEDIC")),
    "english": ("English", ("LATIN",)),
    "gujarati": ("Gujarati", ("GUJARATI",)),
    "hindi": ("Hindi", ("DEVANAGARI", "VEDIC")),
    "kannada": ("Kannada", ("KANNADA",)),
    "khari boli": ("Khari Boli", ("DEVANAGARI", "VEDIC")),
    "malayalam": ("Malayalam", ("MALAYALAM",)),
    "marathi": ("Marathi", ("DEVANAGARI", "VEDIC")),
    "nepali": ("Nepali", ("DEVANAGARI", "VEDIC")),
    "odia": ("Odia", ("ORIYA",)),
    "persian": ("Persian", ("ARABIC",)),
    "punjabi": ("Punjabi", ("GURMUKHI",)),
    "sanskrit": ("Sanskrit", ("DEVANAGARI", "VEDIC")),
    "tamil": ("Tamil", ("TAMIL",)),
    "telugu": ("Telugu", ("TELUGU",)),
    "urdu": ("Urdu", ("ARABIC",)),
}

LANGUAGE_ALIASES = {
    "as": "assamese",
    "awa": "awadhi",
    "bangla": "bengali",
    "bn": "bengali",
    "braj bhasha": "braj",
    "en": "english",
    "fa": "persian",
    "farsi": "persian",
    "gu": "gujarati",
    "hi": "hindi",
    "kn": "kannada",
    "khari-boli": "khari boli",
    "ml": "malayalam",
    "mr": "marathi",
    "ne": "nepali",
    "or": "odia",
    "oriya": "odia",
    "pa": "punjabi",
    "panjabi": "punjabi",
    "sa": "sanskrit",
    "ta": "tamil",
    "te": "telugu",
    "ur": "urdu",
}


def parse_language(value: str) -> str:
    """Normalize a supported book language for script validation."""
    normalized = " ".join(value.strip().casefold().split())
    language = LANGUAGE_ALIASES.get(normalized, normalized)
    if language not in LANGUAGE_SCRIPTS:
        supported = ", ".join(
            display_name
            for display_name, _ in LANGUAGE_SCRIPTS.values()
        )
        raise argparse.ArgumentTypeError(
            f"unsupported language '{value}'. Supported languages: {supported}"
        )
    return language


def _language_details(language: str) -> tuple[str, tuple[str, ...]]:
    """Return the display name and Unicode-name prefixes for a language."""
    try:
        return LANGUAGE_SCRIPTS[language]
    except KeyError as exc:
        raise ValueError(f"Unsupported normalized language '{language}'") from exc


def build_page_batches(
    input_file: Path,
    batch_size: int,
    tmp_dir: Path,
    page_range: tuple[int, int] | None = None,
) -> list[PageBatch]:
    """Build ordered batches from explicit original PDF pages."""
    reader = pypdf.PdfReader(input_file)
    total_pages = len(reader.pages)
    first_page, last_page = page_range or (1, total_pages)
    if last_page > total_pages:
        raise ValueError(
            f"Page {last_page} exceeds PDF length ({total_pages})."
        )

    batches: list[PageBatch] = []
    for batch_start in range(first_page, last_page + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, last_page)
        pages = tuple(
            BatchPage(number=page_number)
            for page_number in range(batch_start, batch_end + 1)
        )
        batch_path = (
            tmp_dir
            / f"{input_file.stem}_batch_{batch_start:04d}-{batch_end:04d}.pdf"
        )

        writer = pypdf.PdfWriter()
        for page in pages:
            writer.add_page(reader.pages[page.number - 1])
        with batch_path.open("wb") as file:
            writer.write(file)

        batches.append(PageBatch(pdf_path=batch_path, pages=pages))

    selected_pages = last_page - first_page + 1
    print(
        f"  Prepared {len(batches)} page batches of up to {batch_size} pages "
        f"({selected_pages} selected pages)"
    )
    return batches


def split_page_batch(batch: PageBatch) -> list[PageBatch]:
    """Write one single-page PDF for every page in an existing page batch."""
    reader = pypdf.PdfReader(batch.pdf_path)
    if len(reader.pages) != len(batch.pages):
        raise ValueError(
            f"{batch.page_label} has {len(batch.pages)} source-page labels "
            f"but its PDF contains {len(reader.pages)} pages"
        )

    single_page_batches = []
    for pdf_page, source_page in zip(reader.pages, batch.pages, strict=True):
        page_path = batch.pdf_path.with_name(
            f"{batch.pdf_path.stem}_page_{source_page.number:04d}.pdf"
        )
        writer = pypdf.PdfWriter()
        writer.add_page(pdf_page)
        with page_path.open("wb") as file:
            writer.write(file)
        single_page_batches.append(
            PageBatch(pdf_path=page_path, pages=(source_page,))
        )
    return single_page_batches


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


def _uses_allowed_script(
    character: str,
    allowed_name_prefixes: tuple[str, ...],
) -> bool:
    """Return whether a Unicode character is neutral or uses an allowed script."""
    category = unicodedata.category(character)
    if category[0] not in {"L", "M"}:
        return True

    unicode_name = unicodedata.name(character, "")
    return any(
        unicode_name.startswith(prefix)
        for prefix in allowed_name_prefixes
    )


@dataclass(frozen=True)
class MarkdownPageSection:
    """Generated Markdown attributed to one original PDF page."""

    page_number: int | None
    text: str
    first_output_line: int


def _markdown_page_sections(markdown_text: str) -> list[MarkdownPageSection]:
    """Split generated Markdown using required source-page markers."""
    matches = list(PAGE_MARKER_PATTERN.finditer(markdown_text))
    if not matches:
        return [
            MarkdownPageSection(
                page_number=None,
                text=markdown_text,
                first_output_line=1,
            )
        ]

    sections: list[MarkdownPageSection] = []
    prefix = markdown_text[:matches[0].start()]
    if prefix.strip():
        sections.append(
            MarkdownPageSection(
                page_number=None,
                text=prefix,
                first_output_line=1,
            )
        )

    for index, match in enumerate(matches):
        content_start = match.end()
        content_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(markdown_text)
        )
        sections.append(
            MarkdownPageSection(
                page_number=int(match.group(1)),
                text=markdown_text[content_start:content_end],
                first_output_line=markdown_text.count(
                    "\n",
                    0,
                    content_start,
                )
                + 1,
            )
        )
    return sections


def validate_page_markers(
    markdown_text: str,
    batch: PageBatch,
) -> list[str]:
    """Require one ordered source-page marker for every page in the batch."""
    found_pages = tuple(
        int(match.group(1))
        for match in PAGE_MARKER_PATTERN.finditer(markdown_text)
    )
    if found_pages == batch.page_numbers:
        return []
    return [
        f"[PAGE_MARKERS][{batch.page_label}] expected "
        f"{batch.page_numbers}, found {found_pages}. Emit exactly one ordered "
        "source-page marker before the content from each page."
    ]


def validate_language(
    markdown_text: str,
    language: str,
    batch: PageBatch,
) -> tuple[list[str], list[str]]:
    """Return non-failing warnings for every foreign-script text run."""
    normalized_language = parse_language(language)
    _, allowed_name_prefixes = _language_details(normalized_language)
    seen_locations: set[tuple[str, int, str]] = set()
    warnings: list[str] = []
    seen_urls: set[tuple[str, int, str]] = set()

    for section in _markdown_page_sections(markdown_text):
        page_label = (
            f"page {section.page_number}"
            if section.page_number is not None
            else batch.page_label
        )
        for section_line, line in enumerate(
            section.text.splitlines(),
            start=0,
        ):
            line_number = section.first_output_line + section_line
            url_spans = []
            for match in URL_PATTERN.finditer(line):
                url = match.group(0).rstrip(".,;:!?।॥")
                if url:
                    url_spans.append(
                        (match.start(), match.start() + len(url), url)
                    )

            run_start: int | None = None
            for index in range(len(line) + 1):
                is_foreign = (
                    index < len(line)
                    and not _uses_allowed_script(
                        line[index],
                        allowed_name_prefixes,
                    )
                )
                if is_foreign and run_start is None:
                    run_start = index
                    continue
                if is_foreign or run_start is None:
                    continue

                foreign_text = line[run_start:index]
                containing_url = next(
                    (
                        url
                        for url_start, url_end, url in url_spans
                        if run_start < url_end and index > url_start
                    ),
                    None,
                )
                if containing_url is not None:
                    url_location = (
                        page_label,
                        line_number,
                        containing_url,
                    )
                    if url_location not in seen_urls:
                        context = line.strip()
                        warnings.append(
                            f"[FOREIGN_LANGUAGE_URL][{page_label}] "
                            f"{containing_url!r}; output line {line_number}; "
                            f"context: {context!r}"
                        )
                        seen_urls.add(url_location)
                    run_start = None
                    continue

                location = (page_label, line_number, foreign_text)
                if location not in seen_locations:
                    context_start = max(0, run_start - 40)
                    context_end = min(len(line), index + 40)
                    context = line[context_start:context_end].strip()
                    warnings.append(
                        f"[FOREIGN_LANGUAGE_TEXT][{page_label}] "
                        f"{foreign_text!r}; output line {line_number}; "
                        f"context: {context!r}"
                    )
                    seen_locations.add(location)
                run_start = None

    return [], warnings


def _validation_prompt(
    markdown_text: str,
    language: str,
    batch: PageBatch,
    check_name: str,
    instructions: str,
) -> str:
    """Build one focused AI validation prompt."""
    display_name, _ = _language_details(language)
    return f"""
You are a focused quality-control reviewer for a PDF-to-Markdown transcription.
Compare the uploaded PDF page batch ({batch.page_label}) with the candidate
Markdown below. The expected source pages are {batch.page_numbers}.
The declared language of the book is {display_name}.
Every page must begin with its exact `<!-- source-page: N -->` metadata marker;
these markers are required boundaries, not explanatory commentary.
Foreign-script text is a non-failing review warning handled separately, whether
it is ordinary prose or part of a URL.
Do not report foreign-script text as an error in this validation check.

Validation check: {check_name}

{instructions}

Return exactly PASS if this check finds no material errors.
Otherwise, return only a concise numbered list of specific, actionable errors.
For every error, identify the source page or nearby text when possible and state
what the next conversion attempt must correct. Do not rewrite the document.

--- BEGIN CANDIDATE MARKDOWN ---
{markdown_text}
--- END CANDIDATE MARKDOWN ---
"""


def _parse_validation_result(result: str | None, check_name: str) -> list[str]:
    """Turn one AI validator response into actionable errors."""
    if not result or not result.strip():
        raise ValueError(f"Empty response from {check_name} batch validator")

    result = result.strip()
    if result.casefold() == "pass":
        return []
    return [f"{check_name}: {result}"]


def _empty_response_error(response: object, request_name: str) -> str:
    """Describe why a Gemini response has no usable text."""
    finish_reasons = []
    for candidate in getattr(response, "candidates", None) or []:
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is None:
            continue
        reason_name = getattr(finish_reason, "name", str(finish_reason))
        finish_reasons.append(reason_name)

    if "PROHIBITED_CONTENT" in finish_reasons:
        return (
            f"{request_name}: Gemini returned PROHIBITED_CONTENT with no "
            f"text. {PROHIBITED_CONTENT_RETRY_CONTEXT}"
        )
    if finish_reasons:
        return (
            f"{request_name}: empty response from Gemini; finish reason(s): "
            + ", ".join(finish_reasons)
        )
    return f"{request_name}: empty response from Gemini"


def _response_has_finish_reason(response: object, reason_name: str) -> bool:
    """Return whether any response candidate ended for the named reason."""
    for candidate in getattr(response, "candidates", None) or []:
        finish_reason = getattr(candidate, "finish_reason", None)
        candidate_reason = getattr(finish_reason, "name", str(finish_reason))
        if candidate_reason == reason_name:
            return True
    return False


def _run_batch_validation_check(
    client: genai.Client,
    uploaded_file: object,
    markdown_text: str,
    model: str,
    language: str,
    batch: PageBatch,
    check_name: str,
    instructions: str,
) -> list[str]:
    """Run one focused batch validation and return its actionable errors."""
    validation_prompt = _validation_prompt(
        markdown_text,
        language,
        batch,
        check_name,
        instructions,
    )
    response = client.models.generate_content(
        model=model,
        contents=[uploaded_file, validation_prompt],
    )
    return _parse_validation_result(response.text, check_name)


def validate_completeness(
    client: genai.Client,
    uploaded_file: object,
    markdown_text: str,
    model: str,
    language: str,
    batch: PageBatch,
) -> list[str]:
    """Check that source content is present once and in the original order."""
    return _run_batch_validation_check(
        client,
        uploaded_file,
        markdown_text,
        model,
        language,
        batch,
        *COMPLETENESS_CHECK,
    )


def validate_transcription_fidelity(
    client: genai.Client,
    uploaded_file: object,
    markdown_text: str,
    model: str,
    language: str,
    batch: PageBatch,
) -> list[str]:
    """Check transcription accuracy and preservation of historical language."""
    return _run_batch_validation_check(
        client,
        uploaded_file,
        markdown_text,
        model,
        language,
        batch,
        *TRANSCRIPTION_FIDELITY_CHECK,
    )


def validate_markdown_output(
    client: genai.Client,
    uploaded_file: object,
    markdown_text: str,
    model: str,
    language: str,
    batch: PageBatch,
) -> list[str]:
    """Check Markdown structure and the raw-output contract."""
    return _run_batch_validation_check(
        client,
        uploaded_file,
        markdown_text,
        model,
        language,
        batch,
        *MARKDOWN_OUTPUT_CHECK,
    )


def validate_batch(
    client: genai.Client,
    uploaded_file: object,
    markdown_text: str,
    model: str,
    language: str,
    batch: PageBatch,
) -> tuple[list[str], list[str]]:
    """Run every focused page-batch validator and aggregate its findings."""
    errors = validate_page_markers(markdown_text, batch)
    language_errors, warnings = validate_language(
        markdown_text,
        language,
        batch,
    )
    errors.extend(language_errors)
    errors.extend(
        validate_completeness(
            client,
            uploaded_file,
            markdown_text,
            model,
            language,
            batch,
        )
    )
    errors.extend(
        validate_transcription_fidelity(
            client,
            uploaded_file,
            markdown_text,
            model,
            language,
            batch,
        )
    )
    errors.extend(
        validate_markdown_output(
            client,
            uploaded_file,
            markdown_text,
            model,
            language,
            batch,
        )
    )
    return errors, warnings


def _conversion_prompt(
    batch: PageBatch,
    language: str,
) -> str:
    """Build the shared PDF-to-Markdown conversion prompt for one page batch."""
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
**Delete** page numbers. Join the text before and after it. Preserve appropriate spacing and punctuation. Do not leave an abrupt break.

**A printed page number is separate from a verse number (e.g., "॥ २ ॥"). Delete printed page numbers, but keep verse numbers attached to the verse they close. The machine-readable source-page markers required below are metadata and must still be emitted; they are not transcriptions of printed page numbers.**

**8. Titles, Subheadings, and Invocations (मंगलाचरण):** - Main book title → # (H1). - Section/Ramayana-type chapters (काण्ड) or major divisions → ## (H2). - A named composition that opens a genuinely distinct section --- a titled hymn, a labelled sub-episode, or the first shift into a new grouping --- → ### (H3). Do NOT give every individual दोहा, सोरठा, or चौपाई its own H3 as it recurs through a continuous passage; a single kand-length section can contain hundreds of each, and heading every one defeats the purpose of a heading hierarchy. Where you still want to flag the type of a recurring verse without a full heading, use a small inline label instead, e.g. 
**(दोहा)**. - Dedications or invocations (e.g., "ॐ","श्री गणेशाय नमः") keep as standalone italic lines using *...*. 
  - Sanskrit framing verses: many Awadhi/Braj devotional works (Tulsidas's Ramcharitmanas is the best-known example) open each major division with several Sanskrit श्लोक --- invocations distinct from the vernacular narrative that follows --- and often close it with a short Sanskrit passage too. If the source does this, keep these shlokas in Sanskrit exactly as scanned, set apart with a label such as "(संस्कृत श्लोक)" or a blockquote, and do NOT apply Rule 2's Braj/Awadhi archaic-verb-form logic to them --- they follow Sanskrit grammar, not Awadhi conjugation, so "correcting" them toward Awadhi forms would be as wrong as modernizing them.

**9. Footnotes / टिप्पणी:** If there are footnotes (marked by *, †, or superscript numbers), convert them to Markdown footnotes ([^1]). Keep footnote identifiers unique across the entire document by numbering them continuously rather than resetting to [^1] in every poem. Most Markdown renderers treat footnote IDs as global to the whole document, so reused identifiers will make later definitions silently overwrite or fail to resolve earlier ones. Still place each footnote's definition at the end of its own poem, separated by a horizontal rule (---), so it stays visually close to its context even though the identifier itself is unique document-wide.

**10. Tables of Contents (अनुक्रमणिका):** If present, convert to a nested Markdown unordered list (- ), preserving indentation levels. Keep the TOC's printed page-number references because they correspond to the source-page markers in the converted document.

**11. Mandatory Page‑by‑Page Processing (No Skipping):** Process every page of the source strictly in order. Do **not** skip, condense, merge, or silently omit any page, even if it seems repetitive, damaged, or hard to read. Transcribe what is legible; for entirely illegible portions, insert a bracketed note such as `[अस्पष्ट]` or `[पृष्ठ X – पूर्णतः अस्पष्ट]` and continue to the next page. If a page is blank, emit its required source-page marker and a line `[रिक्त पृष्ठ]`. At the end of your conversion, verify that every page in the batch has exactly one marker in order.

**Output Format:** 
- Return the converted document as raw Markdown without surrounding code fences or any other wrapper.
- Process every page of the source, strictly in order. Do not skip, condense, merge, or silently omit any page, even one that seems repetitive, damaged, or hard to read --- transcribe what is legible and mark an unclear portion with something like [अस्पष्ट] rather than dropping the page.
- Do not include any explanatory text, summaries, romanization, or feedback. Begin directly with the first required source-page marker, and return nothing except the markers and converted text itself --- no commentary on the conversion process, no sentence noting that a batch is complete, and no note or guess about where a future response should resume."
"""
    normalized_language = parse_language(language)
    display_name, allowed_name_prefixes = _language_details(normalized_language)
    allowed_scripts = ", ".join(allowed_name_prefixes)
    expected_markers = "\n".join(
        f"<!-- source-page: {page.number} -->"
        for page in batch.pages
    )
    prompt += f"""

**Source Page Boundaries (Mandatory):**
- This batch contains these original PDF pages in order: {batch.page_numbers}.
- Before the converted content from each page, emit its exact marker:
{expected_markers}
- Emit every marker exactly once and in that order, including for blank or
  illegible pages. These metadata markers are the only permitted Latin text.

**Declared Book Language (Mandatory):**
- The book language is {display_name}.
- Outside punctuation, numbers, symbols, and whitespace, every
  Unicode letter and combining mark in the conversion must use the declared
  language's script ({allowed_scripts}). The required source-page markers above
  are the sole exception.
- Do not substitute words in any other language or script. When a scan is
  ambiguous, resolve it as {display_name} text using the declared script.
"""
    return prompt


def _conversion_contents(
    uploaded_file: object,
    prompt: str,
    validation_errors: Sequence[str],
) -> list[object]:
    """Build conversion request contents, including prior validation feedback."""
    contents: list[object] = [uploaded_file, prompt]
    if validation_errors:
        formatted_errors = "\n\n".join(
            f"{index}. {error}"
            for index, error in enumerate(validation_errors, start=1)
        )
        contents.append(
            """
The previous conversion attempt failed validation. Correct every issue below
while following all original conversion instructions:

"""
            + formatted_errors
        )
    return contents


def convert_batch(
    client: genai.Client,
    batch: PageBatch,
    batch_index: int,
    total_batches: int,
    model: str,
    language: str,
) -> str:
    """Synchronously convert and validate one page batch with retries."""
    normalized_language = parse_language(language)
    prompt = _conversion_prompt(batch, normalized_language)
    uploaded_file = None
    validation_errors: list[str] = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

            uploaded_file = client.files.upload(file=batch.pdf_path)

            conversion_contents = _conversion_contents(
                uploaded_file,
                prompt,
                validation_errors,
            )

            response = client.models.generate_content(
                model=model,
                contents=conversion_contents,
            )

            text = response.text
            if not text or not text.strip():
                empty_response_error = _empty_response_error(
                    response,
                    "Conversion",
                )
                validation_errors = [empty_response_error]
                raise ValueError(empty_response_error)

            current_validation_errors, current_validation_warnings = validate_batch(
                client,
                uploaded_file,
                text,
                model,
                normalized_language,
                batch,
            )
            if current_validation_errors:
                validation_errors = current_validation_errors
                raise ValueError(
                    "Batch validation failed:\n"
                    + "\n".join(validation_errors)
                )
            for warning in current_validation_warnings:
                print(f"  WARNING {warning}")

            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
            uploaded_file = None

            print(
                f"  Batch {batch_index + 1}/{total_batches} "
                f"({batch.page_label}) converted and validated successfully"
            )
            return text

        except Exception as exc:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass
                uploaded_file = None

            if attempt == MAX_RETRIES:
                print(
                    f"  Batch {batch_index + 1}/{total_batches} "
                    f"({batch.page_label}) FAILED after {MAX_RETRIES} "
                    f"attempts: {exc}"
                )
                raise

            backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
            print(
                f"  Batch {batch_index + 1}/{total_batches} "
                f"({batch.page_label}) attempt {attempt} failed: {exc}"
            )
            print(f"    Retrying in {backoff}s...")
            time.sleep(backoff)

    raise RuntimeError("unreachable retry state")


def _batch_job_state_name(batch_job: object) -> str:
    """Return a stable state name from either an SDK enum or test double."""
    state = getattr(batch_job, "state", None)
    if state is None:
        return "JOB_STATE_UNSPECIFIED"
    return getattr(state, "name", str(state))


def _batch_error_text(error: object | None) -> str:
    """Return the useful message from a Batch API job or request error."""
    if error is None:
        return "unknown Batch API error"
    message = getattr(error, "message", None)
    if message:
        return str(message)
    return str(error)


def _run_batch_api_job(
    client: genai.Client,
    model: str,
    requests: Sequence[types.InlinedRequest],
    display_name: str,
) -> list[types.InlinedResponse]:
    """Submit, poll, and return one inline Gemini Batch API job."""
    if not requests:
        return []

    batch_job = client.batches.create(
        model=model,
        src=list(requests),
        config={"display_name": display_name},
    )
    if not batch_job.name:
        raise RuntimeError("Gemini Batch API returned a job without a name")

    print(
        f"  Created Gemini Batch API job {batch_job.name} "
        f"with {len(requests)} requests"
    )
    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    last_state = ""
    try:
        while _batch_job_state_name(batch_job) not in terminal_states:
            state_name = _batch_job_state_name(batch_job)
            if state_name != last_state:
                print(f"  Batch API job {batch_job.name}: {state_name}")
                last_state = state_name
            time.sleep(BATCH_POLL_INTERVAL)
            batch_job = client.batches.get(name=batch_job.name)
    except KeyboardInterrupt:
        try:
            client.batches.cancel(name=batch_job.name)
            print(f"  Cancelled Gemini Batch API job {batch_job.name}")
        except Exception as cancel_error:
            print(
                f"  WARNING unable to cancel Batch API job {batch_job.name}: "
                f"{cancel_error}"
            )
        raise

    state_name = _batch_job_state_name(batch_job)
    print(f"  Batch API job {batch_job.name}: {state_name}")
    if state_name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(
            f"Gemini Batch API job {batch_job.name} ended in {state_name}: "
            f"{_batch_error_text(getattr(batch_job, 'error', None))}"
        )

    destination = getattr(batch_job, "dest", None)
    responses = (
        getattr(destination, "inlined_responses", None)
        if destination is not None
        else None
    )
    if responses is None:
        raise RuntimeError(
            f"Gemini Batch API job {batch_job.name} returned no inline responses"
        )
    if len(responses) != len(requests):
        raise RuntimeError(
            f"Gemini Batch API job {batch_job.name} returned "
            f"{len(responses)} responses for {len(requests)} requests"
        )
    return list(responses)


def _batch_response_text(
    inline_response: types.InlinedResponse,
    request_name: str,
) -> tuple[str | None, str | None]:
    """Return either response text or a concise per-request Batch API error."""
    error = getattr(inline_response, "error", None)
    if error is not None:
        return None, f"{request_name}: {_batch_error_text(error)}"

    response = getattr(inline_response, "response", None)
    text = getattr(response, "text", None) if response is not None else None
    if not text or not text.strip():
        return None, _empty_response_error(response, request_name)
    return text, None


def _failed_batch_markdown(batch: PageBatch) -> str:
    """Return explicit per-page markers for a batch that exhausted retries."""
    return "\n\n".join(
        f"<!-- source-page: {page.number} -->\n"
        f"<!-- CONVERSION FAILED: page {page.number} -->"
        for page in batch.pages
    )


def convert_batches_with_batch_api(
    client: genai.Client,
    batches: Sequence[PageBatch],
    model: str,
    language: str,
) -> list[str]:
    """Convert and validate page batches through asynchronous Batch API jobs."""
    normalized_language = parse_language(language)
    original_units = [
        ConversionUnit(parent_index=index, batch=batch)
        for index, batch in enumerate(batches)
    ]
    units_by_parent: dict[int, list[ConversionUnit]] = {
        unit.parent_index: [unit] for unit in original_units
    }
    uploaded_files: dict[ConversionUnit, object] = {}
    markdown_results: dict[ConversionUnit, str] = {}
    validation_feedback: dict[ConversionUnit, list[str]] = {
        unit: [] for unit in original_units
    }
    pending = set(original_units)

    def unit_sort_key(unit: ConversionUnit) -> tuple[int, tuple[int, ...]]:
        return unit.parent_index, unit.batch.page_numbers

    def upload_unit(unit: ConversionUnit) -> None:
        uploaded_files[unit] = client.files.upload(file=unit.batch.pdf_path)

    try:
        for unit in original_units:
            print(
                f"  Uploading batch {unit.parent_index + 1}/{len(batches)} "
                f"({unit.batch.page_label})..."
            )
            upload_unit(unit)

        for attempt in range(1, MAX_RETRIES + 1):
            request_units = sorted(pending, key=unit_sort_key)
            conversion_requests = [
                types.InlinedRequest(
                    contents=_conversion_contents(
                        uploaded_files[unit],
                        _conversion_prompt(
                            unit.batch,
                            normalized_language,
                        ),
                        validation_feedback[unit],
                    ),
                    metadata={
                        "stage": "conversion",
                        "batch_index": str(unit.parent_index),
                        "pages": unit.batch.page_label,
                    },
                )
                for unit in request_units
            ]
            conversion_responses = _run_batch_api_job(
                client,
                model,
                conversion_requests,
                f"content-pipeline-conversion-attempt-{attempt}",
            )

            candidates: dict[ConversionUnit, str] = {}
            attempt_errors: dict[ConversionUnit, list[str]] = {}
            prohibited_units: dict[ConversionUnit, str] = {}
            for unit, inline_response in zip(
                request_units,
                conversion_responses,
                strict=True,
            ):
                text, error = _batch_response_text(
                    inline_response,
                    "Conversion",
                )
                if error is not None:
                    response = getattr(inline_response, "response", None)
                    if (
                        len(unit.batch.pages) > 1
                        and _response_has_finish_reason(
                            response,
                            "PROHIBITED_CONTENT",
                        )
                    ):
                        prohibited_units[unit] = error
                    else:
                        attempt_errors[unit] = [error]
                else:
                    assert text is not None
                    candidates[unit] = text

            for unit, error in prohibited_units.items():
                print(
                    f"  Batch {unit.parent_index + 1}/{len(batches)} "
                    f"({unit.batch.page_label}) attempt {attempt} failed:"
                )
                print(f"    {error}")
                single_page_units = [
                    ConversionUnit(
                        parent_index=unit.parent_index,
                        batch=single_page_batch,
                    )
                    for single_page_batch in split_page_batch(unit.batch)
                ]
                units_by_parent[unit.parent_index] = single_page_units
                pending.remove(unit)
                print(
                    f"  Splitting {unit.batch.page_label} into "
                    f"{len(single_page_units)} single-page requests for the "
                    "next Gemini Batch API job."
                )
                for single_page_unit in single_page_units:
                    print(f"    Uploading {single_page_unit.batch.page_label}...")
                    upload_unit(single_page_unit)
                    validation_feedback[single_page_unit] = [error]
                    pending.add(single_page_unit)

            validation_requests: list[types.InlinedRequest] = []
            validation_request_keys: list[tuple[ConversionUnit, str]] = []
            warnings_by_unit: dict[ConversionUnit, list[str]] = {}
            for unit, markdown_text in candidates.items():
                marker_errors = validate_page_markers(
                    markdown_text,
                    unit.batch,
                )
                language_errors, warnings = validate_language(
                    markdown_text,
                    normalized_language,
                    unit.batch,
                )
                attempt_errors[unit] = marker_errors + language_errors
                warnings_by_unit[unit] = warnings

                for check_name, instructions in AI_VALIDATION_CHECKS:
                    validation_requests.append(
                        types.InlinedRequest(
                            contents=[
                                uploaded_files[unit],
                                _validation_prompt(
                                    markdown_text,
                                    normalized_language,
                                    unit.batch,
                                    check_name,
                                    instructions,
                                ),
                            ],
                            metadata={
                                "stage": "validation",
                                "batch_index": str(unit.parent_index),
                                "pages": unit.batch.page_label,
                                "check": check_name,
                            },
                        )
                    )
                    validation_request_keys.append((unit, check_name))

            if validation_requests:
                validation_responses = _run_batch_api_job(
                    client,
                    model,
                    validation_requests,
                    f"content-pipeline-validation-attempt-{attempt}",
                )
                for (unit, check_name), inline_response in zip(
                    validation_request_keys,
                    validation_responses,
                    strict=True,
                ):
                    result, error = _batch_response_text(
                        inline_response,
                        f"{check_name} batch validator",
                    )
                    if error is not None:
                        attempt_errors[unit].append(error)
                        continue
                    try:
                        attempt_errors[unit].extend(
                            _parse_validation_result(result, check_name)
                        )
                    except ValueError as validation_error:
                        attempt_errors[unit].append(str(validation_error))

            for unit in request_units:
                if unit in prohibited_units:
                    continue
                errors = attempt_errors.get(unit, [])
                if errors:
                    validation_feedback[unit] = errors
                    print(
                        f"  Batch {unit.parent_index + 1}/{len(batches)} "
                        f"({unit.batch.page_label}) attempt {attempt} failed:"
                    )
                    for error in errors:
                        print(f"    {error}")
                    continue

                markdown_results[unit] = candidates[unit]
                pending.remove(unit)
                for warning in warnings_by_unit[unit]:
                    print(f"  WARNING {warning}")
                print(
                    f"  Batch {unit.parent_index + 1}/{len(batches)} "
                    f"({unit.batch.page_label}) converted and validated "
                    "successfully"
                )

            if not pending:
                break
            if attempt < MAX_RETRIES:
                backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
                print(
                    f"  Retrying {len(pending)} failed page request(s) "
                    f"in {backoff}s..."
                )
                time.sleep(backoff)

        for unit in sorted(pending, key=unit_sort_key):
            print(
                f"  Batch {unit.parent_index + 1}/{len(batches)} "
                f"({unit.batch.page_label}) FAILED after {MAX_RETRIES} attempts"
            )
            markdown_results[unit] = _failed_batch_markdown(unit.batch)

        assembled_results = []
        for parent_index in range(len(batches)):
            parent_units = sorted(
                units_by_parent[parent_index],
                key=unit_sort_key,
            )
            assembled_results.append(
                "\n\n".join(
                    markdown_results.get(
                        unit,
                        _failed_batch_markdown(unit.batch),
                    )
                    for unit in parent_units
                )
            )
        return assembled_results
    finally:
        for uploaded_file in uploaded_files.values():
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


def convert_pdf(
    input_file: Path,
    output_dir: Path,
    client: genai.Client,
    language: str,
    batch_size: int = BATCH_SIZE,
    model: str = DEFAULT_MODEL,
    page_range: tuple[int, int] | None = None,
    sync: bool = False,
) -> Path:
    """Convert numbered pages with Batch API by default or synchronously."""

    normalized_language = parse_language(language)

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

    if page_range is not None:
        print(f"  Selecting pages {page_range[0]}-{page_range[1]}...")

    with tempfile.TemporaryDirectory() as tmp_name:
        batches = build_page_batches(
            input_file,
            batch_size,
            Path(tmp_name),
            page_range,
        )
        if sync:
            print("  Execution mode: synchronous generate_content API")
            markdown_batches = []
            for batch_index, batch in enumerate(batches):
                if batch_index > 0:
                    print("  Sleeping for 3 seconds between batches...")
                    time.sleep(3)

                print(
                    f"  Processing batch {batch_index + 1}/{len(batches)} "
                    f"({batch.page_label})..."
                )
                try:
                    markdown_batches.append(
                        convert_batch(
                            client,
                            batch,
                            batch_index,
                            len(batches),
                            model,
                            normalized_language,
                        )
                    )
                except Exception:
                    print(
                        f"  Skipping batch {batch_index + 1} "
                        f"({batch.page_label}) due to repeated failures."
                    )
                    markdown_batches.append(_failed_batch_markdown(batch))
        else:
            print("  Execution mode: asynchronous Gemini Batch API")
            markdown_batches = convert_batches_with_batch_api(
                client,
                batches,
                model,
                normalized_language,
            )

        md_text = "\n\n".join(markdown_batches)

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
        "--language",
        type=parse_language,
        required=True,
        metavar="LANGUAGE",
        help="Language of the source book; required for Unicode script validation.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=BATCH_SIZE,
        help=f"Numbered pages per Gemini batch. Defaults to {BATCH_SIZE}.",
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
    parser.add_argument(
        "--sync",
        action="store_true",
        help=(
            "Use synchronous generate_content requests instead of the "
            "default asynchronous Gemini Batch API."
        ),
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
                language=args.language,
                batch_size=args.batch_size,
                model=args.model,
                page_range=args.pages,
                sync=args.sync,
            )
        except Exception as exc:
            failures += 1
            print(f"Error converting '{pdf_file}': {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
