from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pypdf

from content_pipeline.gemini_md import (
    BatchPage,
    PageBatch,
    build_page_batches,
    convert_batch,
    parse_args,
    validate_batch,
    validate_language,
    validate_page_markers,
)


def make_batch(*page_numbers: int) -> PageBatch:
    return PageBatch(
        pdf_path=Path("batch.pdf"),
        pages=tuple(BatchPage(number) for number in page_numbers),
    )


class FakeFiles:
    def __init__(self) -> None:
        self.uploaded: list[Path] = []
        self.deleted: list[str] = []

    def upload(self, *, file: Path) -> object:
        self.uploaded.append(file)
        return SimpleNamespace(name=f"files/batch-{len(self.uploaded)}")

    def delete(self, *, name: str) -> None:
        self.deleted.append(name)


class FakeModels:
    def __init__(self, response_texts: list[str]) -> None:
        self.response_texts = iter(response_texts)
        self.calls: list[dict[str, object]] = []

    def generate_content(self, *, model: str, contents: list[object]) -> object:
        self.calls.append({"model": model, "contents": contents})
        return SimpleNamespace(text=next(self.response_texts))


class FakeClient:
    def __init__(self, response_texts: list[str]) -> None:
        self.files = FakeFiles()
        self.models = FakeModels(response_texts)


class BuildPageBatchesTests(unittest.TestCase):
    def test_batches_retain_original_page_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp_dir = Path(tmp_name)
            input_file = tmp_dir / "book.pdf"
            writer = pypdf.PdfWriter()
            for page_number in range(1, 26):
                writer.add_blank_page(width=page_number, height=100)
            with input_file.open("wb") as file:
                writer.write(file)

            batches = build_page_batches(
                input_file,
                batch_size=5,
                tmp_dir=tmp_dir,
                page_range=(6, 17),
            )

            self.assertEqual(
                [batch.page_numbers for batch in batches],
                [
                    (6, 7, 8, 9, 10),
                    (11, 12, 13, 14, 15),
                    (16, 17),
                ],
            )
            self.assertEqual(
                batches[0].pdf_path.name,
                "book_batch_0006-0010.pdf",
            )
            widths = [
                int(page.mediabox.width)
                for page in pypdf.PdfReader(batches[0].pdf_path).pages
            ]
            self.assertEqual(widths, [6, 7, 8, 9, 10])


class ValidateBatchTests(unittest.TestCase):
    def test_runs_each_focused_check_and_aggregates_errors(self) -> None:
        client = FakeClient(
            [
                "PASS",
                "1. Page 21 reads भूठ but the source reads झूठ.",
                "PASS",
            ]
        )
        batch = make_batch(20, 21)
        markdown = (
            "<!-- source-page: 20 -->\n# प्रत्याशी\n"
            "<!-- source-page: 21 -->\nपाठ"
        )

        errors, warnings = validate_batch(
            client,
            SimpleNamespace(name="files/batch"),
            markdown,
            "test-model",
            "hindi",
            batch,
        )

        self.assertEqual(
            errors,
            [
                "Transcription and orthographic fidelity: "
                "1. Page 21 reads भूठ but the source reads झूठ."
            ],
        )
        self.assertEqual(warnings, [])
        self.assertEqual(len(client.models.calls), 3)
        prompts = [call["contents"][1] for call in client.models.calls]
        self.assertIn("Completeness and ordering", prompts[0])
        self.assertIn("Transcription and orthographic fidelity", prompts[1])
        self.assertIn("Markdown structure and output contract", prompts[2])

    def test_retries_batch_with_exact_page_validation_feedback(self) -> None:
        client = FakeClient(
            [
                "<!-- source-page: 7 -->\nपहला प्रारूप",
                "1. A stanza from page 7 is missing.",
                "PASS",
                "PASS",
                "<!-- source-page: 7 -->\nसुधारा हुआ प्रारूप",
                "PASS",
                "PASS",
                "PASS",
            ]
        )
        batch = make_batch(7)

        with patch("content_pipeline.gemini_md.time.sleep") as sleep:
            result = convert_batch(
                client,
                batch,
                batch_index=0,
                total_batches=1,
                model="test-model",
                language="hindi",
            )

        self.assertEqual(
            result,
            "<!-- source-page: 7 -->\nसुधारा हुआ प्रारूप",
        )
        self.assertEqual(
            client.files.uploaded,
            [Path("batch.pdf"), Path("batch.pdf")],
        )
        self.assertEqual(len(client.models.calls), 8)
        sleep.assert_called_once_with(10)

        retry_contents = client.models.calls[4]["contents"]
        self.assertEqual(len(retry_contents), 3)
        self.assertIn("previous conversion attempt failed validation", retry_contents[2])
        self.assertIn("A stanza from page 7 is missing.", retry_contents[2])

    def test_foreign_text_warning_prints_every_occurrence_without_retrying(
        self,
    ) -> None:
        markdown = (
            "<!-- source-page: 7 -->\n"
            "पहला English प्रारूप और اردو पाठ"
        )
        client = FakeClient([markdown, "PASS", "PASS", "PASS"])
        batch = make_batch(7)
        output = io.StringIO()

        with (
            redirect_stdout(output),
            patch("content_pipeline.gemini_md.time.sleep") as sleep,
        ):
            result = convert_batch(
                client,
                batch,
                batch_index=0,
                total_batches=1,
                model="test-model",
                language="hindi",
            )

        self.assertEqual(result, markdown)
        self.assertEqual(len(client.models.calls), 4)
        sleep.assert_not_called()
        validation_prompts = [
            call["contents"][1]
            for call in client.models.calls[1:]
        ]
        self.assertTrue(
            all(
                "Do not report foreign-script text as an error" in prompt
                for prompt in validation_prompts
            )
        )
        printed = output.getvalue()
        self.assertIn(
            "WARNING [FOREIGN_LANGUAGE_TEXT][page 7] 'English'",
            printed,
        )
        self.assertIn(
            "WARNING [FOREIGN_LANGUAGE_TEXT][page 7] 'اردو'",
            printed,
        )

    def test_url_warning_prints_exact_page_without_retrying(self) -> None:
        markdown = (
            "<!-- source-page: 12 -->\n"
            "अधिक जानकारी: https://www.hindikosh.in"
        )
        client = FakeClient([markdown, "PASS", "PASS", "PASS"])
        batch = make_batch(12)
        output = io.StringIO()

        with (
            redirect_stdout(output),
            patch("content_pipeline.gemini_md.time.sleep") as sleep,
        ):
            result = convert_batch(
                client,
                batch,
                batch_index=0,
                total_batches=1,
                model="test-model",
                language="hindi",
            )

        self.assertEqual(result, markdown)
        self.assertEqual(len(client.models.calls), 4)
        sleep.assert_not_called()
        validation_prompts = [
            call["contents"][1]
            for call in client.models.calls[1:]
        ]
        self.assertTrue(
            all(
                "Do not report foreign-script text as an error" in prompt
                for prompt in validation_prompts
            )
        )
        self.assertIn(
            "[FOREIGN_LANGUAGE_URL][page 12] "
            "'https://www.hindikosh.in'",
            output.getvalue(),
        )


class PageValidationTests(unittest.TestCase):
    def test_reports_foreign_text_on_the_exact_marked_page(self) -> None:
        batch = make_batch(40, 41)
        markdown = (
            "<!-- source-page: 40 -->\n# मानसरोवर\n"
            "<!-- source-page: 41 -->\nयह English और اردو पाठ है। १२३ ₹!"
        )

        errors, warnings = validate_language(markdown, "hindi", batch)
        combined_warnings = "\n".join(warnings)

        self.assertEqual(errors, [])
        self.assertIn("[FOREIGN_LANGUAGE_TEXT][page 41]", combined_warnings)
        self.assertIn("'English'", combined_warnings)
        self.assertIn("'اردو'", combined_warnings)
        self.assertIn("context:", combined_warnings)

    def test_reports_every_foreign_text_occurrence(self) -> None:
        batch = make_batch(8)
        markdown = "<!-- source-page: 8 -->\n" + "\n".join(
            f"पाठ Foreign{index}"
            for index in range(35)
        )

        errors, warnings = validate_language(markdown, "hindi", batch)

        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 35)
        self.assertIn("'Foreign'", warnings[0])
        self.assertIn("context: 'पाठ Foreign0'", warnings[0])
        self.assertIn("context: 'पाठ Foreign34'", warnings[-1])

    def test_allows_declared_script_punctuation_numbers_and_symbols(self) -> None:
        batch = make_batch(3)
        errors, warnings = validate_language(
            "<!-- source-page: 3 -->\n# मानसरोवर\nयह हिन्दी पाठ है — १२३! ₹",
            "Hindi",
            batch,
        )

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_page_marker_validation_requires_every_batch_page_in_order(self) -> None:
        batch = make_batch(3, 4)
        errors = validate_page_markers(
            "<!-- source-page: 4 -->\nपाठ",
            batch,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("[PAGE_MARKERS][pages 3-4]", errors[0])
        self.assertIn("expected (3, 4), found (4,)", errors[0])

    def test_language_and_batch_size_are_parsed_by_the_cli(self) -> None:
        args = parse_args(
            [
                "--language",
                "Hindi",
                "--batch-size",
                "6",
                "--output-dir",
                "out",
                "book.pdf",
            ]
        )
        self.assertEqual(args.language, "hindi")
        self.assertEqual(args.batch_size, 6)

        with (
            patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            parse_args(["--output-dir", "out", "book.pdf"])


if __name__ == "__main__":
    unittest.main()
