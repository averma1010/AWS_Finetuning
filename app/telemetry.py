"""OpenTelemetry setup: traces, metrics, and helper functions."""
from __future__ import annotations
from typing import List, Optional
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

_tracer: Optional[trace.Tracer] = None
_job_created_counter = None
_job_completed_counter = None
_job_failed_counter = None
_job_duration_histogram = None
_dataset_upload_counter = None
_inference_counter = None


def setup_telemetry(app, settings) -> None:
    global _tracer, _job_created_counter, _job_completed_counter
    global _job_failed_counter, _job_duration_histogram
    global _dataset_upload_counter, _inference_counter

    if not settings.otel_enabled:
        return

    resource = Resource.create({"service.name": settings.otel_service_name})

    # --- Tracer ---
    tracer_provider = TracerProvider(resource=resource)
    if settings.otel_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter(endpoint=f"{settings.otel_endpoint}/v1/traces")
    else:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        exporter = ConsoleSpanExporter()
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer(settings.otel_service_name)

    # --- Meter ---
    if settings.otel_endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        metric_exporter = OTLPMetricExporter(endpoint=f"{settings.otel_endpoint}/v1/metrics")
    else:
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
        metric_exporter = ConsoleMetricExporter()

    reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=30_000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)
    meter = metrics.get_meter(settings.otel_service_name)

    _job_created_counter = meter.create_counter(
        "ft.jobs.created", description="Number of finetuning jobs created"
    )
    _job_completed_counter = meter.create_counter(
        "ft.jobs.completed", description="Number of completed finetuning jobs"
    )
    _job_failed_counter = meter.create_counter(
        "ft.jobs.failed", description="Number of failed finetuning jobs"
    )
    _job_duration_histogram = meter.create_histogram(
        "ft.job.duration_seconds", description="Training job duration in seconds", unit="s"
    )
    _dataset_upload_counter = meter.create_counter(
        "ft.datasets.uploaded", description="Number of datasets uploaded"
    )
    _inference_counter = meter.create_counter(
        "ft.inference.requests", description="Number of inference requests"
    )

    FastAPIInstrumentor.instrument_app(app)


def get_tracer() -> trace.Tracer:
    return _tracer or trace.get_tracer("finetuning-service")


def record_job_created(base_model: str, method: str) -> None:
    if _job_created_counter:
        _job_created_counter.add(1, {"base_model": base_model, "method": method})


def record_job_completed(base_model: str, method: str, duration_seconds: Optional[float] = None) -> None:
    if _job_completed_counter:
        _job_completed_counter.add(1, {"base_model": base_model, "method": method})
    if _job_duration_histogram and duration_seconds is not None:
        _job_duration_histogram.record(duration_seconds, {"base_model": base_model, "method": method})


def record_job_failed(base_model: str, method: str) -> None:
    if _job_failed_counter:
        _job_failed_counter.add(1, {"base_model": base_model, "method": method})


def record_dataset_uploaded(num_rows: int) -> None:
    if _dataset_upload_counter:
        _dataset_upload_counter.add(1, {"num_rows_bucket": _bucket(num_rows, [100, 1000, 10000])})


def record_inference_request(base_model: str) -> None:
    if _inference_counter:
        _inference_counter.add(1, {"base_model": base_model})


def _bucket(value: int, boundaries: List[int]) -> str:
    for b in boundaries:
        if value <= b:
            return f"<={b}"
    return f">{boundaries[-1]}"
