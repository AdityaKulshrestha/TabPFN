#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for tabpfn.parallel_execute."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import torch

from tabpfn.parallel_execute import parallel_execute


def test__parallel_execute__single_device__executes_in_current_thread() -> None:
    def test_function(device: torch.device) -> int:  # noqa: ARG001
        return threading.get_ident()

    thread_ids = parallel_execute(
        devices=[torch.device("cpu")], functions=[test_function, test_function]
    )

    current_thread_id = threading.get_ident()
    assert list(thread_ids) == [current_thread_id, current_thread_id]


def test__parallel_execute__single_device__results_in_same_order_as_functions() -> None:
    def a(device: torch.device) -> str:  # noqa: ARG001
        return "a"

    def b(device: torch.device) -> str:  # noqa: ARG001
        return "b"

    def c(device: torch.device) -> str:  # noqa: ARG001
        return "c"

    results = parallel_execute(devices=[torch.device("cpu")], functions=[a, b, c])

    assert list(results) == ["a", "b", "c"]


def test__parallel_execute__multiple_devices__executes_in_worker_threads() -> None:
    def test_function(device: torch.device) -> int:  # noqa: ARG001
        return threading.get_ident()

    thread_ids = parallel_execute(
        devices=[torch.device("cpu"), torch.device("meta")],
        functions=[test_function, test_function],
    )

    current_thread_id = threading.get_ident()
    for thread_id in thread_ids:
        assert thread_id != current_thread_id


def test__parallel_execute__multiple_devices__results_in_same_order_as_functions() -> (
    None
):
    def a(device: torch.device) -> str:  # noqa: ARG001
        return "a"

    def b(device: torch.device) -> str:  # noqa: ARG001
        return "b"

    def c(device: torch.device) -> str:  # noqa: ARG001
        return "c"

    results = parallel_execute(
        devices=[torch.device("meta"), torch.device("meta")], functions=[a, b, c]
    )

    assert list(results) == ["a", "b", "c"]


def test__parallel_execute__xpu_devices__uses_xpu_event_not_cuda_event(
    mocker: MagicMock,
) -> None:
    """Regression test: the xpu branch must use torch.xpu.Event, not torch.cuda.Event.

    A copy-paste from the cuda branch previously left `torch.cuda.Event()` in the
    xpu code path, which segfaulted on machines with no CUDA runtime. See the xpu
    branch of `_execute_function_in_thread`.
    """
    mock_cuda = mocker.patch("torch.cuda")
    mock_xpu = mocker.patch("torch.xpu")

    def test_function(device: torch.device) -> str:  # noqa: ARG001
        return "ok"

    results = parallel_execute(
        devices=[torch.device("xpu:0"), torch.device("xpu:1")],
        functions=[test_function, test_function],
    )

    assert list(results) == ["ok", "ok"]
    mock_xpu.Event.assert_called()
    mock_xpu.device.assert_called()
    mock_cuda.Event.assert_not_called()


def test__parallel_execute__cuda_devices__uses_cuda_event_not_xpu_event(
    mocker: MagicMock,
) -> None:
    """The reverse of the xpu regression test above, for symmetry."""
    mock_cuda = mocker.patch("torch.cuda")
    mock_xpu = mocker.patch("torch.xpu")

    def test_function(device: torch.device) -> str:  # noqa: ARG001
        return "ok"

    results = parallel_execute(
        devices=[torch.device("cuda:0"), torch.device("cuda:1")],
        functions=[test_function, test_function],
    )

    assert list(results) == ["ok", "ok"]
    mock_cuda.Event.assert_called()
    mock_xpu.Event.assert_not_called()
