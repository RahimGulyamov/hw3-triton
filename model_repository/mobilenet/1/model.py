"""MobileNetV3 inference and request-level custom metrics for Triton."""

from pathlib import Path
import os
import time

import numpy as np
import torch
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # Artifacts travel with each Model Analyzer config variant.
        self.model = torch.jit.load(str(Path(__file__).with_name("weights.pt")), map_location="cpu")
        self.model.eval()
        labels = {
            "model": args["model_name"],
            "version": args["model_version"],
            # Python backend 23.12 does not provide an instance-name argument.
            # Each synchronous Python model instance has its own stub process.
            "instance": str(os.getpid()),
        }
        self.time_family = pb_utils.MetricFamily(
            name="hw3_processing_seconds_total",
            description="Sum of processing duration of requests; excludes scheduler queue time",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.active_family = pb_utils.MetricFamily(
            name="hw3_inflight_requests",
            description="Requests executing in the Python backend; excludes queued requests",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.processing = self.time_family.Metric(labels=labels)
        self.inflight = self.active_family.Metric(labels=labels)

    def execute(self, requests):
        count = len(requests)
        if count == 0:
            return []
        started = time.perf_counter()
        self.inflight.set(count)
        try:
            images = [pb_utils.get_input_tensor_by_name(r, "IMAGE").as_numpy() for r in requests]
            sizes = [image.shape[0] for image in images]
            # Combining requests is essential: otherwise dynamic batching has no benefit.
            batch = torch.from_numpy(np.concatenate(images, axis=0))
            with torch.inference_mode():
                logits = self.model(batch).numpy()
            responses = []
            offset = 0
            for size in sizes:
                output = np.ascontiguousarray(logits[offset : offset + size], dtype=np.float32)
                responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("LOGITS", output)]))
                offset += size
            return responses
        except Exception as exc:
            pb_utils.Logger.log_error(f"MobileNet inference failed: {exc}")
            return [pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc))) for _ in requests]
        finally:
            # Each request shares the full batch processing interval. The counter
            # is request-seconds, so concurrent work can exceed wall-clock time.
            try:
                self.processing.increment((time.perf_counter() - started) * count)
            finally:
                self.inflight.set(0)

    def finalize(self):
        # Metric objects must be destroyed before their MetricFamily objects.
        del self.processing
        del self.inflight
        del self.time_family
        del self.active_family
