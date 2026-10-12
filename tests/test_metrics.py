"""Offline failure-path tests. Stubs cover the API boundary, not real Triton."""

import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch


class MetricsFailureTests(unittest.TestCase):
    def setUp(self):
        self.backend = types.ModuleType("triton_python_backend_utils")
        self.backend.get_input_tensor_by_name = Mock(side_effect=RuntimeError("input unavailable"))
        self.backend.Logger = types.SimpleNamespace(log_error=Mock())
        self.backend.TritonError = RuntimeError
        self.backend.InferenceResponse = lambda **kwargs: types.SimpleNamespace(**kwargs)
        path = Path(__file__).resolve().parents[1] / "model_repository/mobilenet/1/model.py"
        spec = importlib.util.spec_from_file_location("hw3_backend", path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {
            "numpy": types.ModuleType("numpy"),
            "torch": types.ModuleType("torch"),
            "triton_python_backend_utils": self.backend,
        }):
            spec.loader.exec_module(self.module)
        self.model = self.module.TritonPythonModel()
        self.model.processing = Mock()
        self.model.inflight = Mock()

    def test_failure_returns_one_error_per_request_and_cleans_gauge(self):
        with patch.object(self.module.time, "perf_counter", side_effect=[10.0, 10.25]):
            responses = self.model.execute([object(), object()])
        self.assertEqual(len(responses), 2)
        self.assertIsNot(responses[0], responses[1])
        self.assertTrue(all(str(response.error) == "input unavailable" for response in responses))
        self.assertEqual([call.args[0] for call in self.model.inflight.set.call_args_list], [2, 0])
        self.model.processing.increment.assert_called_once_with(0.5)

    def test_gauge_resets_even_when_counter_update_raises(self):
        self.model.processing.increment.side_effect = RuntimeError("metric update failed")
        with self.assertRaisesRegex(RuntimeError, "metric update failed"):
            self.model.execute([object()])
        self.model.inflight.set.assert_called_with(0)

    def test_empty_request_list_leaves_metrics_unchanged(self):
        self.assertEqual(self.model.execute([]), [])
        self.model.processing.increment.assert_not_called()
        self.model.inflight.set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
