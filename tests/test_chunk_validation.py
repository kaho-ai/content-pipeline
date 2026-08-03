from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from content_pipeline.gemini_md import (
    convert_chunk,
    parse_args,
    validate_chunk,
    validate_language,
)


class FakeFiles:
    def __init__(self) -> None:
        self.uploaded: list[Path] = []
        self.deleted: list[str] = []

    def upload(self, *, file: Path) -> object:
        self.uploaded.append(file)
        return SimpleNamespace(name=f"files/chunk-{len(self.uploaded)}")

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


class ValidateChunkTests(unittest.TestCase):
    def test_runs_each_focused_check_and_aggregates_errors(self) -> None:
        client = FakeClient(
            [
                "PASS",
                "1. Page 2 reads भूठ but the source reads झूठ.",
                "PASS",
            ]
        )
        uploaded_file = SimpleNamespace(name="files/chunk")

        errors = validate_chunk(
            client,
            uploaded_file,
            "# प्रत्याशी",
            "test-model",
            "hindi",
        )

        self.assertEqual(
            errors,
            [
                "Transcription and orthographic fidelity: "
                "1. Page 2 reads भूठ but the source reads झूठ."
            ],
        )
        self.assertEqual(len(client.models.calls), 3)
        prompts = [call["contents"][1] for call in client.models.calls]
        self.assertIn("Completeness and ordering", prompts[0])
        self.assertIn("Transcription and orthographic fidelity", prompts[1])
        self.assertIn("Markdown structure and output contract", prompts[2])

    def test_retries_conversion_with_aggregated_validation_feedback(self) -> None:
        client = FakeClient(
            [
                "पहला English प्रारूप",
                "1. A stanza from page 2 is missing.",
                "PASS",
                "PASS",
                "सुधारा हुआ प्रारूप",
                "PASS",
                "PASS",
                "PASS",
            ]
        )

        with patch("content_pipeline.gemini_md.time.sleep") as sleep:
            result = convert_chunk(
                client,
                Path("chunk.pdf"),
                chunk_index=0,
                total_chunks=1,
                model="test-model",
                language="hindi",
            )

        self.assertEqual(result, "सुधारा हुआ प्रारूप")
        self.assertEqual(client.files.uploaded, [Path("chunk.pdf"), Path("chunk.pdf")])
        self.assertEqual(len(client.models.calls), 8)
        sleep.assert_called_once_with(10)

        retry_contents = client.models.calls[4]["contents"]
        self.assertEqual(len(retry_contents), 3)
        self.assertIn("previous conversion attempt failed validation", retry_contents[2])
        self.assertIn("foreign-script text 'English'", retry_contents[2])
        self.assertIn("U+0045 LATIN CAPITAL LETTER E", retry_contents[2])
        self.assertIn("Completeness and ordering", retry_contents[2])
        self.assertIn("A stanza from page 2 is missing.", retry_contents[2])


class LanguageValidationTests(unittest.TestCase):
    def test_reports_specific_latin_and_arabic_text_in_hindi_output(self) -> None:
        errors = validate_language(
            "# मानसरोवर\nयह English और اردو पाठ है। १२३ ₹!",
            "hindi",
        )
        combined_errors = "\n".join(errors)

        self.assertIn("line 2", combined_errors)
        self.assertIn("'English'", combined_errors)
        self.assertIn("U+0045 LATIN CAPITAL LETTER E", combined_errors)
        self.assertIn("'اردو'", combined_errors)
        self.assertIn("U+0627 ARABIC LETTER ALEF", combined_errors)
        self.assertIn("Context:", combined_errors)

    def test_allows_declared_script_punctuation_numbers_and_symbols(self) -> None:
        errors = validate_language(
            "# मानसरोवर\nयह हिन्दी पाठ है — १२३! ₹",
            "Hindi",
        )

        self.assertEqual(errors, [])

    def test_language_is_required_and_normalized_by_the_cli(self) -> None:
        args = parse_args(
            [
                "--language",
                "Hindi",
                "--output-dir",
                "out",
                "book.pdf",
            ]
        )
        self.assertEqual(args.language, "hindi")

        with (
            patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            parse_args(["--output-dir", "out", "book.pdf"])


if __name__ == "__main__":
    unittest.main()
