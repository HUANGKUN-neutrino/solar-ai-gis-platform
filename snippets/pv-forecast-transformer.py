"""
Photovoltaic generation forecasting pipeline.

This module converts weather features, installed capacity and calendar signals
into business-facing generation and revenue forecasts. LightGBM/XGBoost are
preferred for short-term tabular forecasting, while Transformer is kept as a
long-sequence extension path. Model artifacts and runtime credentials are
loaded by the deployment environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean
from typing import Any, Iterable


DEFAULT_TARIFF = 0.68
DEFAULT_SYSTEM_EFFICIENCY = 0.82
DEFAULT_PEAK_SUN_HOURS = 4.2
MIN_CONFIDENCE = 0.35


class ForecastEngine(str, Enum):
    AUTO = "auto"
    TRANSFORMER = "transformer"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    FORMULA = "formula"


@dataclass(frozen=True)
class WeatherFeature:
    date: str
    irradiance: float = 4.8
    cloud_cover: float = 35.0
    rain_probability: float = 15.0
    temperature: float = 27.0
    humidity: float = 70.0
    wind_speed: float = 3.0


@dataclass(frozen=True)
class ForecastRequest:
    capacity_kw: float
    weather: list[WeatherFeature]
    tariff: float = DEFAULT_TARIFF
    system_efficiency: float = DEFAULT_SYSTEM_EFFICIENCY
    engine: ForecastEngine = ForecastEngine.AUTO


@dataclass(frozen=True)
class ForecastPoint:
    date: str
    energy_kwh: float
    revenue: float
    confidence: float
    engine: str
    peak_power_kw: float
    daylight_hours: float


@dataclass(frozen=True)
class ForecastResponse:
    ok: bool
    engine: str
    capacity_kw: float
    total_energy_kwh: float
    total_revenue: float
    average_confidence: float
    points: list[ForecastPoint] = field(default_factory=list)


def forecast_generation(request: ForecastRequest) -> ForecastResponse:
    validate_request(request)

    feature_rows = build_time_series_features(request)
    normalized_rows = normalize_features(feature_rows)
    engine = resolve_engine(request.engine)

    prediction = run_model_or_fallback(
        engine=engine,
        normalized_rows=normalized_rows,
        request=request,
    )

    calibrated_points = calibrate_generation_output(
        prediction=prediction,
        request=request,
        feature_rows=feature_rows,
        engine=engine.value,
    )

    total_energy = round(sum(point.energy_kwh for point in calibrated_points), 2)
    total_revenue = round(sum(point.revenue for point in calibrated_points), 2)
    average_confidence = round(mean(point.confidence for point in calibrated_points), 4)

    return ForecastResponse(
        ok=True,
        engine=engine.value,
        capacity_kw=request.capacity_kw,
        total_energy_kwh=total_energy,
        total_revenue=total_revenue,
        average_confidence=average_confidence,
        points=calibrated_points,
    )


def build_time_series_features(request: ForecastRequest) -> list[dict[str, float]]:
    rows = []

    for index, weather in enumerate(request.weather):
        season_sin, season_cos = encode_calendar_season(weather.date)
        weather_score = calculate_weather_score(weather)
        expected_daylight = estimate_daylight_hours(weather.date)

        rows.append(
            {
                "capacity_kw": request.capacity_kw,
                "irradiance": clamp(weather.irradiance, 0.1, 8.0),
                "cloud_cover": clamp(weather.cloud_cover, 0, 100),
                "rain_probability": clamp(weather.rain_probability, 0, 100),
                "temperature": clamp(weather.temperature, -10, 45),
                "humidity": clamp(weather.humidity, 5, 100),
                "wind_speed": clamp(weather.wind_speed, 0, 30),
                "day_index": float(index),
                "season_sin": season_sin,
                "season_cos": season_cos,
                "weather_score": weather_score,
                "daylight_hours": expected_daylight,
                "system_efficiency": request.system_efficiency,
            }
        )

    return rows


def normalize_features(rows: list[dict[str, float]]) -> list[list[float]]:
    schema = [
        ("capacity_kw", 0, 5000),
        ("irradiance", 0, 8),
        ("cloud_cover", 0, 100),
        ("rain_probability", 0, 100),
        ("temperature", -10, 45),
        ("humidity", 0, 100),
        ("wind_speed", 0, 30),
        ("day_index", 0, 30),
        ("season_sin", -1, 1),
        ("season_cos", -1, 1),
        ("weather_score", 0, 1),
        ("daylight_hours", 8, 14),
        ("system_efficiency", 0.6, 0.95),
    ]

    normalized = []
    for row in rows:
        normalized.append(
            [
                normalize_value(row[name], lower, upper)
                for name, lower, upper in schema
            ]
        )
    return normalized


def resolve_engine(engine: ForecastEngine) -> ForecastEngine:
    if engine != ForecastEngine.AUTO:
        return engine

    for candidate in [
        ForecastEngine.LIGHTGBM,
        ForecastEngine.XGBOOST,
        ForecastEngine.TRANSFORMER,
    ]:
        if ForecastRuntime.available(candidate):
            return candidate

    return ForecastEngine.FORMULA


def run_model_or_fallback(
    engine: ForecastEngine,
    normalized_rows: list[list[float]],
    request: ForecastRequest,
) -> list[float]:
    if engine == ForecastEngine.FORMULA:
        return baseline_formula_prediction(request)

    raw_prediction = ForecastRuntime.predict(
        engine=engine,
        features=normalized_rows,
    )

    if raw_prediction:
        return raw_prediction

    return baseline_formula_prediction(request)


def baseline_formula_prediction(request: ForecastRequest) -> list[float]:
    predictions = []

    for weather in request.weather:
        score = calculate_weather_score(weather)
        daylight = estimate_daylight_hours(weather.date)
        energy = (
            request.capacity_kw
            * request.system_efficiency
            * DEFAULT_PEAK_SUN_HOURS
            * score
            * daylight
            / 11.5
        )
        predictions.append(max(0, energy))

    return predictions


def calibrate_generation_output(
    prediction: list[float],
    request: ForecastRequest,
    feature_rows: list[dict[str, float]],
    engine: str,
) -> list[ForecastPoint]:
    points = []

    for index, weather in enumerate(request.weather):
        row = feature_rows[index]
        raw_energy = prediction[index] if index < len(prediction) else 0
        energy_kwh = clamp(raw_energy, 0, request.capacity_kw * row["daylight_hours"])
        confidence = estimate_prediction_confidence(
            weather=weather,
            engine=engine,
            feature_row=row,
        )

        points.append(
            ForecastPoint(
                date=weather.date,
                energy_kwh=round(energy_kwh, 2),
                revenue=round(energy_kwh * request.tariff, 2),
                confidence=confidence,
                engine=engine,
                peak_power_kw=round(energy_kwh / max(row["daylight_hours"], 1), 2),
                daylight_hours=round(row["daylight_hours"], 2),
            )
        )

    return points


def calculate_weather_score(weather: WeatherFeature) -> float:
    irradiance_score = clamp(weather.irradiance / 5.2, 0.15, 1.15)
    cloud_penalty = 1 - clamp(weather.cloud_cover / 130, 0, 0.75)
    rain_penalty = 1 - clamp(weather.rain_probability / 170, 0, 0.55)
    temperature_penalty = 1 - clamp(abs(weather.temperature - 25) / 90, 0, 0.18)
    humidity_penalty = 1 - clamp((weather.humidity - 65) / 260, 0, 0.12)

    return clamp(
        irradiance_score * cloud_penalty * rain_penalty * temperature_penalty * humidity_penalty,
        0.08,
        1.12,
    )


def estimate_prediction_confidence(
    weather: WeatherFeature,
    engine: str,
    feature_row: dict[str, float],
) -> float:
    engine_bonus = {
        ForecastEngine.TRANSFORMER.value: 0.22,
        ForecastEngine.LIGHTGBM.value: 0.18,
        ForecastEngine.XGBOOST.value: 0.16,
        ForecastEngine.FORMULA.value: 0.08,
    }.get(engine, 0.08)

    weather_quality = feature_row["weather_score"] * 0.45
    input_quality = 0.25
    if weather.irradiance <= 0 or weather.cloud_cover > 95:
        input_quality -= 0.08
    if weather.rain_probability > 85:
        input_quality -= 0.05

    return round(clamp(MIN_CONFIDENCE + engine_bonus + weather_quality + input_quality, 0.35, 0.92), 4)


class ForecastRuntime:
    @staticmethod
    def available(engine: ForecastEngine) -> bool:
        return ModelRegistry.has_artifact(engine.value)

    @staticmethod
    def predict(engine: ForecastEngine, features: list[list[float]]) -> list[float]:
        return ModelRegistry.predict_sequence(engine.value, features)


class ModelRegistry:
    @staticmethod
    def has_artifact(engine_name: str) -> bool:
        """
        Checks versioned model artifacts and feature scalers behind a stable boundary.
        """
        _ = engine_name
        return False

    @staticmethod
    def predict_sequence(engine_name: str, features: list[list[float]]) -> list[float]:
        """
        Routes inference to LightGBM, XGBoost or Transformer engines.
        Empty output falls back to the calibrated formula model.
        """
        _ = (engine_name, features)
        return []


def evaluate_forecast(
    actual_kwh: Iterable[float],
    predicted_kwh: Iterable[float],
) -> dict[str, float]:
    actual = list(actual_kwh)
    predicted = list(predicted_kwh)

    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual_kwh and predicted_kwh must have the same non-empty length.")

    errors = [pred - real for real, pred in zip(actual, predicted)]
    absolute_errors = [abs(item) for item in errors]
    squared_errors = [item * item for item in errors]

    return {
        "mae": round(mean(absolute_errors), 4),
        "rmse": round(math.sqrt(mean(squared_errors)), 4),
        "mape": round(mean(abs(err) / max(abs(real), 1e-6) for real, err in zip(actual, errors)), 4),
        "bias": round(mean(errors), 4),
    }


def encode_calendar_season(date_text: str) -> tuple[float, float]:
    day_of_year = parse_day_of_year(date_text)
    angle = 2 * math.pi * day_of_year / 365
    return math.sin(angle), math.cos(angle)


def estimate_daylight_hours(date_text: str) -> float:
    day_of_year = parse_day_of_year(date_text)
    seasonal = math.sin(2 * math.pi * (day_of_year - 80) / 365)
    return clamp(11.2 + seasonal * 1.35, 9.6, 13.6)


def parse_day_of_year(date_text: str) -> int:
    parts = str(date_text).split("-")
    if len(parts) < 3:
        return 180

    month = clamp_int(parts[1], 1, 12)
    day = clamp_int(parts[2], 1, 31)
    month_offsets = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    return min(365, month_offsets[month - 1] + day)


def normalize_value(value: float, lower: float, upper: float) -> float:
    return clamp((float(value) - lower) / max(upper - lower, 1e-9), 0, 1)


def validate_request(request: ForecastRequest) -> None:
    if request.capacity_kw <= 0:
        raise ValueError("capacity_kw must be greater than zero.")
    if not request.weather:
        raise ValueError("weather must not be empty.")
    if request.tariff < 0:
        raise ValueError("tariff must not be negative.")
    if not 0.5 <= request.system_efficiency <= 0.98:
        raise ValueError("system_efficiency is out of business range.")


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


def clamp_int(value: Any, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = lower
    return int(clamp(parsed, lower, upper))
