"""Convert a PDF to Markdown with Marker.

Usage:
    python -m extraction.convert_pdf <pdf_path> <out_dir> [--name <basename>]

Writes:
    <out_dir>/<basename>.md               Markdown (equations + tables preserved)
    <out_dir>/<basename>/*.png etc.       Extracted figures / images
    <out_dir>/<basename>_meta.json        Marker metadata (page count, toc, etc.)

The output basename defaults to the PDF's stem.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a PDF to Markdown via Marker")
    parser.add_argument("pdf", type=pathlib.Path, help="Input PDF file")
    parser.add_argument("out_dir", type=pathlib.Path, help="Output directory")
    parser.add_argument(
        "--name",
        default=None,
        help="Base filename for outputs (default: PDF stem)",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        print(f"error: PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    basename = args.name or args.pdf.stem

    # Tuning for an 8 GB consumer GPU (e.g. RTX 4060 / 4060 Laptop).
    # Surya's CUDA defaults (recognition batch = 256, foundation encoder chunk
    # = 32768) assume a data-center GPU and blow past 8 GB of VRAM. On Windows
    # / WDDM this does not OOM — it silently spills into shared system RAM
    # over PCIe, which makes text recognition ~10–100x slower (this is the
    # ~25 s/it "Recognizing Text" crawl we observed). The values below cap
    # peak VRAM around 6 GB on an 8 GB card while keeping the GPU busy.
    # Every value uses setdefault so the shell can still override it.
    import os
    os.environ.setdefault("TORCH_DEVICE", "cuda")
    # Recognition batch (text lines processed together): CUDA default None -> 16.
    os.environ.setdefault("RECOGNITION_BATCH_SIZE", "16")
    # Foundation encoder chunk size: CUDA default 32768 -> 8192.
    os.environ.setdefault("FOUNDATION_CHUNK_SIZE", "8192")
    # Lower image DPI for non-OCR stages (detection, layout): 96 -> 64.
    # Each dimension shrinks ~33%, so total pixel count drops ~55%.
    os.environ.setdefault("IMAGE_DPI", "64")
    # Lower image DPI for OCR: 192 -> 128.
    # The high-res pass normally doubles the low-res DPI.
    os.environ.setdefault("IMAGE_DPI_HIGHRES", "128")
    # Other batch sizes.
    os.environ.setdefault("DETECTOR_BATCH_SIZE", "8")
    os.environ.setdefault("LAYOUT_BATCH_SIZE", "4")
    os.environ.setdefault("TABLE_REC_BATCH_SIZE", "4")
    os.environ.setdefault("OCR_ERROR_BATCH_SIZE", "4")
    # Cap the max cached allocator block size on Windows, where
    # expandable_segments is unavailable.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:512",
    )

    # Import Marker lazily so --help stays fast even without the models loaded.
    from marker.converters.pdf import PdfConverter
    from marker.output import save_output

    # Build model dict with a SHARED FoundationPredictor.
    # create_model_dict() loads two separate copies of the foundation model
    # (one for layout, one for recognition), doubling VRAM for the largest
    # model.  Sharing the instance halves that overhead.
    from surya.foundation import FoundationPredictor  # noqa: PLC0415
    from surya.layout import LayoutPredictor  # noqa: PLC0415
    from surya.recognition import RecognitionPredictor  # noqa: PLC0415
    from surya.table_rec import TableRecPredictor  # noqa: PLC0415
    from surya.detection import DetectionPredictor  # noqa: PLC0415
    from surya.ocr_error import OCRErrorPredictor  # noqa: PLC0415
    from surya.settings import settings as surya_settings  # noqa: PLC0415

    shared_fp = FoundationPredictor(checkpoint=surya_settings.LAYOUT_MODEL_CHECKPOINT)
    artifact_dict = {
        "layout_model": LayoutPredictor(shared_fp),
        "recognition_model": RecognitionPredictor(shared_fp),
        "table_rec_model": TableRecPredictor(),
        "detection_model": DetectionPredictor(),
        "ocr_error_model": OCRErrorPredictor(),
    }

    converter = PdfConverter(artifact_dict=artifact_dict)
    import torch  # noqa: PLC0415
    torch.cuda.empty_cache()
    rendered = converter(str(args.pdf))
    save_output(rendered, str(args.out_dir), basename)

    md_path = args.out_dir / f"{basename}.md"
    peak = torch.cuda.max_memory_allocated(0) / (1024**3)
    print(f"ok: wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
