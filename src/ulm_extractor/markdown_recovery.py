"""Conservatively turn PP-StructureV3 Markdown into OCR-complete, text-only Markdown.

Keeps PP's existing headings and HTML/Markdown tables. Replaces image
placeholders with independently recognized text, and appends unresolved OCR
lines in reading order rather than inventing table cells or losing information.
"""
from html import unescape
from pathlib import Path
import re

FORMAT = "ppstructurev3-markdown-recovered-v2"
RECOVERY = {
    "fallback": "full-image-paddleocr-en",
    "image_regions": "pp-layout-box-or-markdown-crop",
    "unplaced": "append-ocr-transcript",
}

IMAGE = re.compile(
    r'<div\b[^>]*>\s*(?P<tag><img\b[^>]*>)\s*</div>|(?P<single><img\b[^>]*>)',
    flags=re.IGNORECASE | re.DOTALL,
)
SRC = re.compile(r'\bsrc\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
TAG = re.compile(r"<[^>]+>")
WORD = re.compile(r"[^\w]+", re.UNICODE)


def _value(raw):
    value = raw.json if hasattr(raw, "json") else raw
    if callable(value):
        value = value()
    if isinstance(value, str):
        import json
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("OCR/parser result must provide a JSON mapping")
    result = value.get("res", value)
    if not isinstance(result, dict):
        raise TypeError("Invalid OCR/parser result")
    return result


def _bbox(item):
    if item is None:
        return None
    if hasattr(item, "tolist"):
        item = item.tolist()
    if len(item) == 4 and all(isinstance(x, (float, int)) for x in item):
        return [float(x) for x in item]
    if len(item) >= 4 and all(hasattr(x, "__len__") and len(x) == 2 for x in item):
        return [float(min(pt[0] for pt in item)),
                float(min(pt[1] for pt in item)),
                float(max(pt[0] for pt in item)),
                float(max(pt[1] for pt in item))]
    return None


def ocr_lines(raw, nested=None, source="fallback"):
    """Return text and xyxy boxes from flat PaddleOCR or PP global OCR."""
    value = _value(raw)
    if nested:
        value = value.get(nested) or {}
    texts = value.get("rec_texts") or []
    boxes = value.get("rec_boxes")
    if boxes is None:
        boxes = value.get("rec_polys")
    if boxes is None:
        boxes = [None] * len(texts)
    if len(texts) != len(boxes):
        raise ValueError("OCR recognized texts and coordinates do not align")
    return [{"text": str(txt).strip(), "bbox": _bbox(box), "source": source}
            for txt, box in zip(texts, boxes) if str(txt).strip()]


def layout_blocks(raw):
    blocks = _value(raw).get("parsing_res_list") or []
    return [{"label": str(b.get("block_label", "")).lower(),
             "bbox": _bbox(b.get("block_bbox")),
             "content": str(b.get("block_content") or ""),
             "order": b.get("block_order")}
            for b in blocks if isinstance(b, dict)]


def _normalized(text):
    # Strip HTML tags without losing visible text from existing HTML tables.
    return " ".join(WORD.sub(" ", unescape(TAG.sub(" ", text))).casefold().split())


def _present(markdown, text):
    line = _normalized(text)
    if not line:
        return True
    return f" {line} " in f" {_normalized(markdown)} "


def _contains(box, candidate):
    if box is None or candidate is None:
        return False
    x = (candidate[0] + candidate[2]) / 2
    y = (candidate[1] + candidate[3]) / 2
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _reading_order(lines):
    # OCR engine output already orders lines, but explicit positions help if
    # a fallback OCR result is ordered differently.
    return sorted(lines, key=lambda l: (
        l["bbox"][1] if l["bbox"] else float("inf"),
        l["bbox"][0] if l["bbox"] else float("inf")))


def _deduplicate(fallback, internal):
    """Prefer the existing English PaddleOCR recognizer when boxes overlap."""
    combined = list(fallback)
    for item in internal:
        if item["bbox"] and any(
            other["bbox"] and
            _contains(other["bbox"], item["bbox"]) and
            _contains(item["bbox"], other["bbox"])
            for other in fallback
        ):
            continue
        if any(other["text"] == item["text"] and other["bbox"] == item["bbox"]
               for other in combined):
            continue
        combined.append(item)
    return _reading_order(combined)


def resolve_markdown(markdown, parser_result, flat_ocr_lines, image_ocr=None):
    """Return (resolved Markdown, diagnostic counts).

    image_ocr maps embedded Markdown image paths to OCR lines from those image
    crops. Only actual OCR predictions are written to the Markdown, never
    guessed field values. Unplaceable text is retained at the end with a clear
    heading; exact cell placement cannot be guaranteed by box geometry.
    """
    image_ocr = image_ocr or {}
    blocks = layout_blocks(parser_result)
    internal = ocr_lines(parser_result, nested="overall_ocr_res", source="pp-internal")
    lines = _deduplicate(flat_ocr_lines, internal)
    image_blocks = [b for b in blocks if "image" in b["label"] or
                    b["label"] in {"figure", "chart"}]
    placeholders = list(IMAGE.finditer(markdown))
    used = set()
    n_image = 0
    n_crop = 0

    def replace_placeholder(match):
        nonlocal n_image, n_crop
        tag = match.group("tag") or match.group("single")
        found = SRC.search(tag)
        src = found.group(1) if found else ""
        region = next((b for b in image_blocks if src and src in b["content"]), None)
        if region is None and len(image_blocks) == len(placeholders):
            region = image_blocks[placeholders.index(match)]
        recovered = []
        if region and region["bbox"]:
            for i, line in enumerate(lines):
                if i not in used and _contains(region["bbox"], line["bbox"]) and not _present(markdown, line["text"]):
                    recovered.append(line["text"])
                    used.add(i)
                    n_image += 1
        crops = image_ocr.get(src, [])
        if not crops:
            crops = image_ocr.get(Path(src).name, [])
        for line in crops:
            if not _present(markdown, line["text"]) and not any(
                _normalized(line["text"]) == _normalized(t) for t in recovered
            ):
                recovered.append(line["text"])
                n_crop += 1
        # Broken local image references are never handed to the text-only LM.
        return "\n" + "\n".join(recovered) + "\n" if recovered else "\n"

    modified = IMAGE.sub(replace_placeholder, markdown).strip()
    unseen = []
    for i, line in enumerate(lines):
        if i in used or _present(modified, line["text"]):
            continue
        # Recovered lines already in the image replacement are not repeated.
        if not any(_normalized(line["text"]) == _normalized(t) for t in unseen):
            unseen.append(line["text"])
    if unseen:
        heading = "### Additional OCR-recognized text (reading order)"
        modified += ("\n\n" if modified else "") + heading + "\n\n" + "\n".join(unseen)
    if not modified.strip():
        raise ValueError("No readable OCR text or Markdown was produced")
    return modified.strip() + "\n", {
        "embedded_image_references": len(placeholders),
        "text_lines_from_image_regions": n_image,
        "text_lines_from_image_crops": n_crop,
        "additional_ocr_lines": len(unseen),
        "fallback_ocr_lines": len(flat_ocr_lines),
        "pp_internal_ocr_lines": len(internal),
    }
