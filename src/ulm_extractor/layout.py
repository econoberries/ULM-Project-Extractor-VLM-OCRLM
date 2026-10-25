"""Serialize PaddleOCR text and detected pixel boxes as LM-readable JSON.

This preserves PP-OCRv6 coordinates; it is not PP-StructureV3 table parsing.
"""
import json

FORMAT = "paddleocr-layout-v1"


def _as_result(output):
    value = output.json if hasattr(output, "json") else output
    if callable(value):
        value = value()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("Expected a PaddleOCR JSON result")
    value = value.get("res", value)
    if not isinstance(value, dict):
        raise ValueError("PaddleOCR result missing res mapping")
    return value


def format_layout_json(outputs, image_size):
    """Preserve every line with xyxy pixel coordinates and confidence."""
    width, height = image_size
    lines = []
    for output in outputs:
        result = _as_result(output)
        texts = result.get("rec_texts")
        boxes = result.get("rec_boxes")
        scores = result.get("rec_scores")
        if not isinstance(texts, list) or boxes is None:
            raise ValueError("PaddleOCR layout requires rec_texts and rec_boxes")
        if len(texts) != len(boxes):
            raise ValueError("OCR texts and bounding boxes have different lengths")
        if scores is not None and len(scores) != len(texts):
            raise ValueError("OCR text/score alignment failed")
        for i, text in enumerate(texts):
            box = boxes[i]
            if len(box) != 4:
                raise ValueError("Each rec_box must have four xyxy coordinates")
            xyxy = [int(v) for v in box]
            if xyxy[2] < xyxy[0] or xyxy[3] < xyxy[1]:
                raise ValueError("Invalid xyxy bounding box")
            lines.append({
                "text": str(text),
                "bbox": xyxy,
                "confidence": float(scores[i]) if scores is not None else None,
            })
    return json.dumps({
        "format": FORMAT,
        "coordinate_system": "absolute_pixels",
        "box_format": "xyxy",
        "image_size": {"width": int(width), "height": int(height)},
        "lines": lines,
    }, ensure_ascii=False, separators=(",", ":"))
