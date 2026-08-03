from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pypdf

from content_pipeline.gemini_md import convert_pdf, parse_args, parse_page_range


class PageRangeTests(unittest.TestCase):
    def test_parses_single_page(self) -> None:
        self.assertEqual(parse_page_range("3"), (3, 3))

    def test_parses_inclusive_range(self) -> None:
        self.assertEqual(parse_page_range("20-25"), (20, 25))

    def test_rejects_descending_range(self) -> None:
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "Start page must be <= end page",
        ):
            parse_page_range("25-20")

    def test_cli_accepts_range(self) -> None:
        args = parse_args(
            [
                "--language",
                "Hindi",
                "--output-dir",
                "out",
                "--pages",
                "20-25",
                "book.pdf",
            ]
        )
        self.assertEqual(args.pages, (20, 25))
        self.assertEqual(args.language, "hindi")


class ConvertPageRangeTests(unittest.TestCase):
    def test_converts_only_selected_pages_in_page_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp_dir = Path(tmp_name)
            input_file = tmp_dir / "book.pdf"
            output_dir = tmp_dir / "output"
            writer = pypdf.PdfWriter()
            for page_number in range(1, 31):
                writer.add_blank_page(width=page_number, height=100)
            with input_file.open("wb") as file:
                writer.write(file)

            converted_batches: list[tuple[str, tuple[int, ...], list[int]]] = []

            def fake_convert_batches(client, batches, model, language):
                for batch in batches:
                    pdf_pages = pypdf.PdfReader(batch.pdf_path).pages
                    converted_batches.append(
                        (
                            batch.pdf_path.name,
                            batch.page_numbers,
                            [
                                int(page.mediabox.width)
                                for page in pdf_pages
                            ],
                        )
                    )
                return [
                    f"batch {batch_index + 1}"
                    for batch_index in range(len(batches))
                ]

            with patch(
                "content_pipeline.gemini_md.convert_batches_with_batch_api",
                side_effect=fake_convert_batches,
            ):
                output_file = convert_pdf(
                    input_file,
                    output_dir,
                    client=object(),
                    language="hindi",
                    batch_size=4,
                    page_range=(20, 25),
                )

            self.assertEqual(output_file.name, "book_pages_20-25.md")
            self.assertEqual(
                converted_batches,
                [
                    (
                        "book_batch_0020-0023.pdf",
                        (20, 21, 22, 23),
                        [20, 21, 22, 23],
                    ),
                    (
                        "book_batch_0024-0025.pdf",
                        (24, 25),
                        [24, 25],
                    ),
                ],
            )
            self.assertEqual(
                output_file.read_text(encoding="utf-8"),
                "batch 1\n\nbatch 2",
            )

    def test_rejects_range_past_end_of_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp_dir = Path(tmp_name)
            input_file = tmp_dir / "book.pdf"
            writer = pypdf.PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with input_file.open("wb") as file:
                writer.write(file)

            with self.assertRaisesRegex(ValueError, "exceeds PDF length"):
                convert_pdf(
                    input_file,
                    tmp_dir / "output",
                    client=object(),
                    language="hindi",
                    page_range=(1, 2),
                )


if __name__ == "__main__":
    unittest.main()
