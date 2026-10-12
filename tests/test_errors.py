#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the error helpers in `tabpfn.errors`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.errors import (
    TabPFNCUDAOutOfMemoryError,
    TabPFNMPSOutOfMemoryError,
    TabPFNXPUOutOfMemoryError,
    handle_oom_errors,
)


class TestHandleOomErrors:
    """The OOM message describes the data the model actually ran on.

    Both counts come off the `X` handed to the context manager, never off
    `n_features_in_`: date expansion can make the frame the transformer runs on
    wider than the one the caller passed, and memory follows the wider one.
    """

    devices = (torch.device("cpu"),)

    def _raise_oom(self, X: np.ndarray) -> str:
        with (
            pytest.raises(TabPFNCUDAOutOfMemoryError) as caught,
            handle_oom_errors(
                self.devices, X, model_type="classifier", n_train_samples=100
            ),
        ):
            raise torch.OutOfMemoryError
        return str(caught.value)

    def test__oom_on_2d_input__message_names_both_sizes(self) -> None:
        message = self._raise_oom(np.zeros((7, 13)))

        assert "100 train / 7 test samples, 13 features." in message

    def test__oom_on_1d_input__omits_the_feature_count(self) -> None:
        message = self._raise_oom(np.zeros(7))

        assert "7 test samples." in message
        assert "features" not in message

    def test__non_oom_runtime_error__propagates_unchanged(self) -> None:
        with (
            pytest.raises(RuntimeError, match="something else"),
            handle_oom_errors(self.devices, np.zeros((2, 2)), model_type="classifier"),
        ):
            raise RuntimeError("something else")


class TestHandleOomErrorsDeviceDispatch:
    """`torch.OutOfMemoryError` is dispatched by `devices[0].type`.

    XPU's allocator raises the same typed `torch.OutOfMemoryError` CUDA does
    (confirmed against the xpu caching allocator), so it must be routed to
    `TabPFNXPUOutOfMemoryError`, not silently fall back to the CUDA one.
    """

    def test__xpu_device__typed_oom__raises_xpu_error(self) -> None:
        with (
            pytest.raises(TabPFNXPUOutOfMemoryError),
            handle_oom_errors(
                (torch.device("xpu:0"),), np.zeros((2, 2)), model_type="classifier"
            ),
        ):
            raise torch.OutOfMemoryError

    def test__cuda_device__typed_oom__raises_cuda_error(self) -> None:
        with (
            pytest.raises(TabPFNCUDAOutOfMemoryError),
            handle_oom_errors(
                (torch.device("cuda:0"),), np.zeros((2, 2)), model_type="classifier"
            ),
        ):
            raise torch.OutOfMemoryError

    def test__unmapped_device__typed_oom__falls_back_to_cuda_error(self) -> None:
        """Unknown/unmapped device types keep the historical CUDA default."""
        with (
            pytest.raises(TabPFNCUDAOutOfMemoryError),
            handle_oom_errors(
                (torch.device("cpu"),), np.zeros((2, 2)), model_type="classifier"
            ),
        ):
            raise torch.OutOfMemoryError

    def test__mps_device__generic_runtime_error__raises_mps_error(self) -> None:
        """MPS raises a plain RuntimeError (no typed OutOfMemoryError), unlike xpu."""
        with (
            pytest.raises(TabPFNMPSOutOfMemoryError),
            handle_oom_errors(
                (torch.device("mps"),), np.zeros((2, 2)), model_type="classifier"
            ),
        ):
            raise RuntimeError("MPS backend out of memory")
