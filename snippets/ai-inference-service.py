"""
AI inference service for roof and photovoltaic detection.

The snippet focuses on service contract, input validation, model runtime routing,
post-processing and business result normalization. Model artifacts, credentials,
business records and deployment configuration are intentionally excluded.
"""

from __future__ import annotations

import base64
import ipaddress
import socket
import struct
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator


MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE_HOSTS = {
    "server.arcgisonline.com",
    "services.arcgisonline.com",
    "tiles.example-map-provider.com",
}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(
    title="Solar Roof AI Inference Service",
    version="1.0.0",
    description="Sanitized service contract for photovoltaic roof detection.",
)


class VisionEngine(str, Enum):
    AUTO = "auto"
    YOLO11 = "yolo11"
    UNET = "unet"
    ONNX = "onnx"


class RoofDetectRequest(BaseModel):
    image_url: str | None = Field(default=None, description="Satellite tile or object storage URL.")
    data_url: str | None = Field(default=None, description="Base64 encoded image for controlled uploads.")
    lat: float | None = None
    lng: float | None = None
    zoom: int | None = Field(default=None, ge=1, le=22)
    engine: VisionEngine = VisionEngine.AUTO
    min_confidence: float = Field(default=0.55, ge=0.05, le=0.95)

    @model_validator(mode="after")
    def require_image_source(self) -> "RoofDetectRequest":
        if not self.image_url and not self.data_url:
            raise ValueError("image_url or data_url is required.")
        if self.image_url and self.data_url:
            raise ValueError("Only one image source can be provided.")
        return self


class DetectionBox(BaseModel):
    label: str
    confidence: float
    bbox: list[float] = Field(description="[x1, y1, x2, y2] in image pixels.")
    area_ratio: float = Field(default=0)


class RoofDetectResponse(BaseModel):
    ok: bool
    engine: str
    elapsed_ms: int
    pv_count: int
    roof_count: int
    roof_area_ratio: float
    pv_status: str
    confidence: float
    review_required: bool
    detections: list[DetectionBox]


@dataclass(frozen=True)
class RawDetection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


@app.get("/api/ai/health")
def health_check() -> dict[str, Any]:
    runtime = get_model_runtime()
    return {
        "ok": True,
        "service": "roof-vision",
        "engines": runtime.available_engines(),
        "defaultEngine": runtime.default_engine,
    }


@app.post("/api/ai/roof-detect", response_model=RoofDetectResponse)
def detect_roof_resource(request: RoofDetectRequest) -> RoofDetectResponse:
    started = time.perf_counter()

    image_bytes = load_image_bytes(request)
    image_meta = inspect_image(image_bytes)
    runtime = get_model_runtime()

    raw_detections = runtime.predict(
        image_bytes=image_bytes,
        image_meta=image_meta,
        engine=request.engine,
    )

    detections = normalize_detections(
        raw_detections,
        image_width=image_meta["width"],
        image_height=image_meta["height"],
        min_confidence=request.min_confidence,
    )

    summary = summarize_roof_resource(detections)

    return RoofDetectResponse(
        ok=True,
        engine=runtime.resolved_engine(request.engine),
        elapsed_ms=round((time.perf_counter() - started) * 1000),
        pv_count=summary["pv_count"],
        roof_count=summary["roof_count"],
        roof_area_ratio=summary["roof_area_ratio"],
        pv_status=summary["pv_status"],
        confidence=summary["confidence"],
        review_required=summary["review_required"],
        detections=detections,
    )


def load_image_bytes(request: RoofDetectRequest) -> bytes:
    if request.data_url:
        return decode_data_url(request.data_url)

    safe_url = validate_remote_image_url(request.image_url or "")
    return fetch_remote_image(safe_url)


def validate_remote_image_url(image_url: str) -> str:
    parsed = urllib.parse.urlparse(image_url)

    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Invalid image_url.")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="Credentials in image_url are not allowed.")
    if parsed.hostname.lower() not in ALLOWED_IMAGE_HOSTS:
        raise HTTPException(status_code=400, detail="image_url host is not allowed.")

    assert_public_dns_target(parsed.hostname)
    return urllib.parse.urlunparse(parsed)


def assert_public_dns_target(hostname: str) -> None:
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail="Cannot resolve image host.") from exc

    for item in addr_info:
        ip = ipaddress.ip_address(item[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise HTTPException(status_code=400, detail="image host resolves to a restricted network.")


def fetch_remote_image(image_url: str) -> bytes:
    request = urllib.request.Request(
        image_url,
        headers={"User-Agent": "solar-roof-ai/1.0"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=6) as response:
            content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
            if content_type and content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(status_code=400, detail="Unsupported remote image type.")
            return read_limited_response(response)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Remote image fetch failed.") from exc


def read_limited_response(response: Any) -> bytes:
    chunks = []
    total = 0

    while True:
        chunk = response.read(256 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Remote image is too large.")
        chunks.append(chunk)

    return b"".join(chunks)


def decode_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise HTTPException(status_code=400, detail="Invalid data_url.")

    header, payload = data_url.split(",", 1)
    media_type = header.split(";")[0].replace("data:", "").lower()

    if media_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image type.")

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload.") from exc

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image payload is too large.")

    return image_bytes


def inspect_image(image_bytes: bytes) -> dict[str, int]:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image payload is empty.")

    return ImageMetadataAdapter.inspect(image_bytes)


class ImageMetadataAdapter:
    @staticmethod
    def inspect(image_bytes: bytes) -> dict[str, int]:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") and len(image_bytes) >= 24:
            width, height = struct.unpack(">II", image_bytes[16:24])
            return {"width": width, "height": height, "bytes": len(image_bytes)}

        jpeg_size = ImageMetadataAdapter._inspect_jpeg(image_bytes)
        if jpeg_size:
            width, height = jpeg_size
            return {"width": width, "height": height, "bytes": len(image_bytes)}

        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return {"width": 1024, "height": 1024, "bytes": len(image_bytes)}

        raise HTTPException(status_code=400, detail="Image payload cannot be decoded.")

    @staticmethod
    def _inspect_jpeg(image_bytes: bytes) -> tuple[int, int] | None:
        if not image_bytes.startswith(b"\xff\xd8"):
            return None

        index = 2
        while index < len(image_bytes) - 9:
            if image_bytes[index] != 0xFF:
                index += 1
                continue

            marker = image_bytes[index + 1]
            block_length = int.from_bytes(image_bytes[index + 2 : index + 4], "big")
            if block_length < 2:
                return None

            if marker in {0xC0, 0xC2}:
                height = int.from_bytes(image_bytes[index + 5 : index + 7], "big")
                width = int.from_bytes(image_bytes[index + 7 : index + 9], "big")
                return width, height

            index += 2 + block_length

        return None


class ModelRuntime:
    default_engine = VisionEngine.YOLO11.value

    def __init__(self) -> None:
        self._engines = {
            VisionEngine.YOLO11: "object-detection-runtime",
            VisionEngine.UNET: "segmentation-runtime",
            VisionEngine.ONNX: "onnx-runtime",
        }

    def available_engines(self) -> list[str]:
        return [engine.value for engine in self._engines]

    def resolved_engine(self, requested: VisionEngine) -> str:
        if requested == VisionEngine.AUTO:
            return self.default_engine
        return requested.value

    def predict(
        self,
        image_bytes: bytes,
        image_meta: dict[str, int],
        engine: VisionEngine,
    ) -> list[RawDetection]:
        resolved_engine = self.resolved_engine(engine)

        if resolved_engine == VisionEngine.YOLO11.value:
            return self._predict_with_detector(image_bytes, image_meta)
        if resolved_engine == VisionEngine.UNET.value:
            return self._predict_with_segmenter(image_bytes, image_meta)
        if resolved_engine == VisionEngine.ONNX.value:
            return self._predict_with_onnx(image_bytes, image_meta)

        raise HTTPException(status_code=400, detail="Unsupported inference engine.")

    def _predict_with_detector(
        self,
        image_bytes: bytes,
        image_meta: dict[str, int],
    ) -> list[RawDetection]:
        return ModelAdapter.predict_detector(image_bytes, image_meta)

    def _predict_with_segmenter(
        self,
        image_bytes: bytes,
        image_meta: dict[str, int],
    ) -> list[RawDetection]:
        return ModelAdapter.predict_segmenter(image_bytes, image_meta)

    def _predict_with_onnx(
        self,
        image_bytes: bytes,
        image_meta: dict[str, int],
    ) -> list[RawDetection]:
        return ModelAdapter.predict_onnx(image_bytes, image_meta)


class ModelAdapter:
    @staticmethod
    def predict_detector(image_bytes: bytes, image_meta: dict[str, int]) -> list[RawDetection]:
        rows = RuntimeRegistry.invoke("roof-detector", image_bytes, image_meta)
        return [coerce_raw_detection(row) for row in rows]

    @staticmethod
    def predict_segmenter(image_bytes: bytes, image_meta: dict[str, int]) -> list[RawDetection]:
        rows = RuntimeRegistry.invoke("roof-segmenter", image_bytes, image_meta)
        return [coerce_raw_detection(row) for row in rows]

    @staticmethod
    def predict_onnx(image_bytes: bytes, image_meta: dict[str, int]) -> list[RawDetection]:
        rows = RuntimeRegistry.invoke("onnx-roof-runtime", image_bytes, image_meta)
        return [coerce_raw_detection(row) for row in rows]


class RuntimeRegistry:
    @staticmethod
    def invoke(
        runtime_name: str,
        image_bytes: bytes,
        image_meta: dict[str, int],
    ) -> list[dict[str, Any]]:
        """
        Runtime adapters connect this boundary to YOLO, U-Net or ONNX sessions.
        Model loading stays behind the registry so the API layer remains stable.
        """
        _ = (runtime_name, image_bytes, image_meta)
        return []


def coerce_raw_detection(row: dict[str, Any]) -> RawDetection:
    bbox = row.get("bbox") or [0, 0, 0, 0]
    if len(bbox) != 4:
        raise HTTPException(status_code=500, detail="Invalid model bbox output.")

    return RawDetection(
        label=str(row.get("label") or "unknown"),
        confidence=float(row.get("confidence") or 0),
        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
    )


def normalize_detections(
    raw_detections: Iterable[RawDetection],
    image_width: int,
    image_height: int,
    min_confidence: float,
) -> list[DetectionBox]:
    normalized = []

    for item in raw_detections:
        if item.confidence < min_confidence:
            continue

        x1, y1, x2, y2 = clamp_bbox(item.bbox, image_width, image_height)
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(image_width * image_height, 1)

        normalized.append(
            DetectionBox(
                label=normalize_label(item.label),
                confidence=round(float(item.confidence), 4),
                bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                area_ratio=round(area_ratio, 6),
            )
        )

    return sorted(normalized, key=lambda item: item.confidence, reverse=True)


def summarize_roof_resource(detections: list[DetectionBox]) -> dict[str, Any]:
    pv_items = [item for item in detections if item.label == "pv_panel"]
    roof_items = [item for item in detections if item.label in {"roof", "available_roof"}]
    confidence = max([item.confidence for item in detections], default=0)
    roof_area_ratio = round(sum(item.area_ratio for item in roof_items), 6)

    if pv_items:
        pv_status = "existing_pv"
    elif roof_items:
        pv_status = "available_roof"
    else:
        pv_status = "unknown"

    return {
        "pv_count": len(pv_items),
        "roof_count": len(roof_items),
        "roof_area_ratio": roof_area_ratio,
        "pv_status": pv_status,
        "confidence": round(confidence, 4),
        "review_required": confidence < 0.7 or pv_status == "unknown",
    }


def normalize_label(label: str) -> str:
    text = label.strip().lower().replace("-", "_").replace(" ", "_")
    label_map = {
        "solar_panel": "pv_panel",
        "photovoltaic": "pv_panel",
        "pv": "pv_panel",
        "building_roof": "roof",
        "industrial_roof": "available_roof",
    }
    return label_map.get(text, text)


def clamp_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        clamp(x1, 0, width),
        clamp(y1, 0, height),
        clamp(x2, 0, width),
        clamp(y2, 0, height),
    )


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def get_model_runtime() -> ModelRuntime:
    return ModelRuntime()
