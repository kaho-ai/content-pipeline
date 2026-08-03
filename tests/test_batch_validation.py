from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pypdf
from google.genai import types as genai_types

from content_pipeline.gemini_md import (
    BATCH_POLL_INTERVAL,
    BatchPage,
    PageBatch,
    _run_batch_api_job,
    build_page_batches,
    convert_batch,
    convert_batches_with_batch_api,
    convert_pdf,
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
        file_number = len(self.uploaded)
        return genai_types.File(
            name=f"files/batch-{file_number}",
            uri=f"https://example.invalid/files/batch-{file_number}",
            mime_type="application/pdf",
        )

    def delete(self, *, name: str) -> None:
        self.deleted.append(name)


class FakeModels:
    def __init__(self, response_texts: list[str]) -> None:
        self.response_texts = iter(response_texts)
        self.calls: list[dict[str, object]] = []

    def generate_content(self, *, model: str, contents: list[object]) -> object:
        self.calls.append({"model": model, "contents": contents})
        return SimpleNamespace(text=next(self.response_texts))


class FakeBatches:
    def __init__(
        self,
        response_sets: list[
            list[str | Exception | genai_types.GenerateContentResponse]
        ],
    ) -> None:
        self.response_sets = iter(response_sets)
        self.calls: list[dict[str, object]] = []
        self.cancelled: list[str] = []

    def create(
        self,
        *,
        model: str,
        src: list[genai_types.InlinedRequest],
        config: dict[str, str],
    ) -> object:
        self.calls.append({"model": model, "src": src, "config": config})
        response_values = next(self.response_sets)
        if len(response_values) != len(src):
            raise AssertionError(
                f"fake has {len(response_values)} responses for {len(src)} requests"
            )
        responses = []
        for value in response_values:
            if isinstance(value, Exception):
                responses.append(
                    genai_types.InlinedResponse(
                        error=genai_types.JobError(message=str(value))
                    )
                )
            elif isinstance(value, genai_types.GenerateContentResponse):
                responses.append(
                    genai_types.InlinedResponse(response=value)
                )
            else:
                responses.append(
                    genai_types.InlinedResponse(
                        response=genai_types.GenerateContentResponse(
                            candidates=[
                                genai_types.Candidate(
                                    content=genai_types.Content(
                                        parts=[genai_types.Part(text=value)]
                                    )
                                )
                            ]
                        )
                    )
                )
        return SimpleNamespace(
            name=f"batches/job-{len(self.calls)}",
            state=SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
            dest=SimpleNamespace(inlined_responses=responses),
            error=None,
        )

    def get(self, *, name: str) -> object:
        raise AssertionError(f"terminal fake job {name} should not be polled")

    def cancel(self, *, name: str) -> None:
        self.cancelled.append(name)


class PollingFakeBatches:
    def __init__(self, *, interrupt: bool = False) -> None:
        self.interrupt = interrupt
        self.get_calls: list[str] = []
        self.cancelled: list[str] = []

    def create(self, **_: object) -> object:
        return SimpleNamespace(
            name="batches/polling-job",
            state=SimpleNamespace(name="JOB_STATE_PENDING"),
            dest=None,
            error=None,
        )

    def get(self, *, name: str) -> object:
        self.get_calls.append(name)
        if self.interrupt:
            raise KeyboardInterrupt
        response = genai_types.InlinedResponse(
            response=genai_types.GenerateContentResponse(
                candidates=[
                    genai_types.Candidate(
                        content=genai_types.Content(
                            parts=[genai_types.Part(text="done")]
                        )
                    )
                ]
            )
        )
        return SimpleNamespace(
            name=name,
            state=SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
            dest=SimpleNamespace(inlined_responses=[response]),
            error=None,
        )

    def cancel(self, *, name: str) -> None:
        self.cancelled.append(name)


class FakeClient:
    def __init__(
        self,
        response_texts: list[str],
        batch_response_sets: list[
            list[str | Exception | genai_types.GenerateContentResponse]
        ]
        | None = None,
    ) -> None:
        self.files = FakeFiles()
        self.models = FakeModels(response_texts)
        self.batches = FakeBatches(batch_response_sets or [])


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


class AsyncBatchApiTests(unittest.TestCase):
    def test_prohibited_multi_page_batch_splits_into_single_page_requests(
        self,
    ) -> None:
        prohibited_response = genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.PROHIBITED_CONTENT
                )
            ]
        )
        page_markdown = [
            f"<!-- source-page: {page_number} -->\nपृष्ठ {page_number}"
            for page_number in (21, 22, 23)
        ]
        client = FakeClient(
            [],
            [
                [prohibited_response],
                page_markdown,
                ["PASS"] * 9,
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_name:
            batch_path = Path(tmp_name) / "batch_0021-0023.pdf"
            writer = pypdf.PdfWriter()
            for page_number in (21, 22, 23):
                writer.add_blank_page(width=page_number, height=100)
            with batch_path.open("wb") as file:
                writer.write(file)
            batch = PageBatch(
                pdf_path=batch_path,
                pages=tuple(
                    BatchPage(page_number)
                    for page_number in (21, 22, 23)
                ),
            )

            with patch("content_pipeline.gemini_md.time.sleep") as sleep:
                results = convert_batches_with_batch_api(
                    client,
                    [batch],
                    "test-model",
                    "hindi",
                )

        self.assertEqual(results, ["\n\n".join(page_markdown)])
        self.assertEqual(
            [len(call["src"]) for call in client.batches.calls],
            [1, 3, 9],
        )
        split_requests = client.batches.calls[1]["src"]
        self.assertEqual(
            [request.metadata["pages"] for request in split_requests],
            ["page 21", "page 22", "page 23"],
        )
        self.assertTrue(
            all(
                "public-domain literary text" in request.contents[2]
                for request in split_requests
            )
        )
        self.assertEqual(len(client.files.uploaded), 4)
        self.assertEqual(len(client.files.deleted), 4)
        sleep.assert_called_once_with(10)

    def test_polls_until_batch_job_finishes(self) -> None:
        client = FakeClient([])
        client.batches = PollingFakeBatches()
        request = genai_types.InlinedRequest(contents="test")

        with patch("content_pipeline.gemini_md.time.sleep") as sleep:
            responses = _run_batch_api_job(
                client,
                "test-model",
                [request],
                "test-job",
            )

        self.assertEqual(responses[0].response.text, "done")
        sleep.assert_called_once_with(BATCH_POLL_INTERVAL)
        self.assertEqual(
            client.batches.get_calls,
            ["batches/polling-job"],
        )

    def test_keyboard_interrupt_cancels_active_remote_job(self) -> None:
        client = FakeClient([])
        client.batches = PollingFakeBatches(interrupt=True)
        request = genai_types.InlinedRequest(contents="test")

        with (
            patch("content_pipeline.gemini_md.time.sleep"),
            self.assertRaises(KeyboardInterrupt),
        ):
            _run_batch_api_job(
                client,
                "test-model",
                [request],
                "test-job",
            )

        self.assertEqual(
            client.batches.cancelled,
            ["batches/polling-job"],
        )

    def test_batches_conversion_and_validation_and_retries_only_failures(
        self,
    ) -> None:
        first_page = "<!-- source-page: 1 -->\nपहला प्रारूप"
        second_page = "<!-- source-page: 2 -->\nदूसरा प्रारूप"
        corrected_first_page = "<!-- source-page: 1 -->\nसुधारा हुआ प्रारूप"
        client = FakeClient(
            [],
            [
                [first_page, second_page],
                [
                    "1. Page 1 is missing a heading.",
                    "PASS",
                    "PASS",
                    "PASS",
                    "PASS",
                    "PASS",
                ],
                [corrected_first_page],
                ["PASS", "PASS", "PASS"],
            ],
        )

        with patch("content_pipeline.gemini_md.time.sleep") as sleep:
            results = convert_batches_with_batch_api(
                client,
                [make_batch(1), make_batch(2)],
                "test-model",
                "hindi",
            )

        self.assertEqual(results, [corrected_first_page, second_page])
        self.assertEqual(client.models.calls, [])
        self.assertEqual(
            [len(call["src"]) for call in client.batches.calls],
            [2, 6, 1, 3],
        )
        sleep.assert_called_once_with(10)
        retry_request = client.batches.calls[2]["src"][0]
        self.assertEqual(retry_request.metadata["batch_index"], "0")
        self.assertIn(
            "Page 1 is missing a heading.",
            retry_request.contents[2],
        )
        self.assertEqual(
            client.files.deleted,
            ["files/batch-1", "files/batch-2"],
        )

    def test_batch_request_error_retries_without_retrying_successes(self) -> None:
        first_page = "<!-- source-page: 1 -->\nपहला प्रारूप"
        second_page = "<!-- source-page: 2 -->\nदूसरा प्रारूप"
        client = FakeClient(
            [],
            [
                [RuntimeError("temporary capacity error"), second_page],
                ["PASS", "PASS", "PASS"],
                [first_page],
                ["PASS", "PASS", "PASS"],
            ],
        )

        with patch("content_pipeline.gemini_md.time.sleep"):
            results = convert_batches_with_batch_api(
                client,
                [make_batch(1), make_batch(2)],
                "test-model",
                "hindi",
            )

        self.assertEqual(results, [first_page, second_page])
        self.assertEqual(
            [len(call["src"]) for call in client.batches.calls],
            [2, 3, 1, 3],
        )

    def test_prohibited_content_retry_explains_public_domain_context(self) -> None:
        prohibited_response = genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.PROHIBITED_CONTENT
                )
            ]
        )
        markdown = "<!-- source-page: 21 -->\nसाधारण हिन्दी पाठ"
        client = FakeClient(
            [],
            [
                [prohibited_response],
                [markdown],
                ["PASS", "PASS", "PASS"],
            ],
        )

        with patch("content_pipeline.gemini_md.time.sleep"):
            results = convert_batches_with_batch_api(
                client,
                [make_batch(21)],
                "test-model",
                "hindi",
            )

        self.assertEqual(results, [markdown])
        retry_request = client.batches.calls[1]["src"][0]
        retry_feedback = retry_request.contents[2]
        self.assertIn("PROHIBITED_CONTENT", retry_feedback)
        self.assertIn("public-domain literary text", retry_feedback)
        self.assertIn("flagged incorrectly", retry_feedback)
        self.assertIn("archival purposes", retry_feedback)

    def test_convert_pdf_defaults_to_batch_api_and_sync_overrides_it(self) -> None:
        markdown = "<!-- source-page: 1 -->\n[रिक्त पृष्ठ]"
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp_dir = Path(tmp_name)
            input_file = tmp_dir / "book.pdf"
            output_dir = tmp_dir / "out"
            writer = pypdf.PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with input_file.open("wb") as file:
                writer.write(file)

            async_client = FakeClient(
                [],
                [[markdown], ["PASS", "PASS", "PASS"]],
            )
            with redirect_stdout(io.StringIO()):
                async_output = convert_pdf(
                    input_file,
                    output_dir,
                    async_client,
                    "hindi",
                    batch_size=1,
                )

            self.assertEqual(async_output.read_text(encoding="utf-8"), markdown)
            self.assertEqual(async_client.models.calls, [])
            self.assertEqual(len(async_client.batches.calls), 2)

            sync_client = FakeClient(
                [markdown, "PASS", "PASS", "PASS"],
            )
            with redirect_stdout(io.StringIO()):
                sync_output = convert_pdf(
                    input_file,
                    output_dir,
                    sync_client,
                    "hindi",
                    batch_size=1,
                    sync=True,
                )

            self.assertEqual(sync_output.read_text(encoding="utf-8"), markdown)
            self.assertEqual(len(sync_client.models.calls), 4)
            self.assertEqual(sync_client.batches.calls, [])


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
        self.assertFalse(args.sync)

        sync_args = parse_args(
            [
                "--language",
                "Hindi",
                "--sync",
                "--output-dir",
                "out",
                "book.pdf",
            ]
        )
        self.assertTrue(sync_args.sync)

        with (
            patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            parse_args(["--output-dir", "out", "book.pdf"])


if __name__ == "__main__":
    unittest.main()
