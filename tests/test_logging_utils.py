"""Tests for sciantist.logging_utils module."""

from __future__ import annotations

from pathlib import Path

import pytest

from sciantist.logging_utils import (
    _inject_worker_id,
    _resolve_worker_id_from_process_name,
    _write_per_worker_log,
    configure_logging,
)


class TestResolveWorkerIdFromProcessName:
    """Test _resolve_worker_id_from_process_name function."""

    def test_merge_worker_pattern(self) -> None:
        result = _resolve_worker_id_from_process_name("sciantist-merge_worker_05")
        assert result == "worker_05"

    def test_sciantist_pattern(self) -> None:
        result = _resolve_worker_id_from_process_name("sciantist-12")
        assert result == "worker_12"

    def test_unknown_returns_main(self) -> None:
        result = _resolve_worker_id_from_process_name("MainProcess")
        assert result == "main"

    def test_empty_string_returns_main(self) -> None:
        result = _resolve_worker_id_from_process_name("")
        assert result == "main"


class TestInjectWorkerId:
    """Test _inject_worker_id function."""

    def test_injects_worker_id(self) -> None:
        record = {"extra": {}}
        _inject_worker_id(record)
        assert "worker_id" in record["extra"]

    def test_injects_main_for_main_process(self) -> None:
        record = {"extra": {}}
        _inject_worker_id(record)
        assert record["extra"]["worker_id"] == "main"


class TestWritePerWorkerLog:
    """Test _write_per_worker_log function."""

    def test_writes_to_worker_log_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import multiprocessing
        monkeypatch.setattr(multiprocessing, "current_process", lambda: type("MockProcess", (), {"name": "worker_01"})())
        message = type("MockMessage", (), {"record": {"extra": {"worker_id": "worker_01"}}})()
        _write_per_worker_log(tmp_path, message)
        log_path = tmp_path / "workers" / "worker_01" / "output.log"
        assert log_path.exists()

    def test_skips_non_worker_messages(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import multiprocessing
        monkeypatch.setattr(multiprocessing, "current_process", lambda: type("MockProcess", (), {"name": "MainProcess"})())
        message = type("MockMessage", (), {"record": {"extra": {"worker_id": "main"}}})()
        _write_per_worker_log(tmp_path, message)
        assert not (tmp_path / "workers").exists()


class TestConfigureLogging:
    """Test configure_logging function."""

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        configure_logging(tmp_path, verbose=False)
        assert tmp_path.exists()

    def test_creates_log_file(self, tmp_path: Path) -> None:
        log_path = configure_logging(tmp_path, verbose=False)
        assert log_path.name == "output.log"

    def test_verbose_sets_debug_level(self, tmp_path: Path) -> None:
        log_path = configure_logging(tmp_path, verbose=True)
        assert log_path.exists()