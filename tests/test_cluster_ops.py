"""Tests for sciantist.cluster_ops module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sciantist.cluster_ops import (
    ClusterProfile,
    _coerce_cluster_profile,
    _parse_lsf_duration_to_seconds,
    _parse_slurm_duration_to_seconds,
    _runtime_to_seconds,
    _wrap_remote_login_shell,
    configure_active_cluster,
    get_active_cluster_profile,
    load_cluster_catalog_yaml,
    load_remote_log,
    submit_cluster_job,
    get_slurm_job_status,
    get_lsf_job_status,
    get_cluster_job_status,
    get_slurm_job_runtime,
    get_lsf_job_runtime,
    get_cluster_job_runtime,
)


class TestRuntimeToSeconds:
    """Test _runtime_to_seconds function."""

    def test_valid_hh_mm_ss(self) -> None:
        assert _runtime_to_seconds("01:30:45") == 5445

    def test_zero_time(self) -> None:
        assert _runtime_to_seconds("00:00:00") == 0

    def test_max_time(self) -> None:
        assert _runtime_to_seconds("23:59:59") == 86399

    def test_invalid_format_not_three_parts(self) -> None:
        with pytest.raises(ValueError, match="HH:MM:SS"):
            _runtime_to_seconds("1:30")

    def test_non_numeric_parts_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            _runtime_to_seconds("ab:cd:ef")


class TestCoerceClusterProfile:
    """Test _coerce_cluster_profile function."""

    def test_minimal_dict(self) -> None:
        result = _coerce_cluster_profile("test", {"scheduler": "slurm"})
        assert result.name == "test"
        assert result.scheduler == "slurm"
        assert result.wandb_sync is True

    def test_full_dict(self) -> None:
        raw = {
            "scheduler": "lsf",
            "cluster_target": "custom-target",
            "ssh_target": "user@host",
            "info_log_path": "/path/{log_id}.out",
            "error_log_path": "/path/{log_id}.err",
            "submit_extra_args": "-q default",
            "wandb_sync": False,
        }
        result = _coerce_cluster_profile("test", raw)
        assert result.cluster_target == "custom-target"
        assert result.ssh_target == "user@host"
        assert result.wandb_sync is False

    def test_invalid_scheduler_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid scheduler"):
            _coerce_cluster_profile("test", {"scheduler": "pbs"})

    def test_invalid_wandb_sync_string_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid wandb_sync"):
            _coerce_cluster_profile("test", {"scheduler": "slurm", "wandb_sync": "maybe"})

    def test_wandb_sync_truthy_strings(self) -> None:
        for truthy in ["1", "true", "yes", "on"]:
            result = _coerce_cluster_profile("test", {"scheduler": "slurm", "wandb_sync": truthy})
            assert result.wandb_sync is True

    def test_wandb_sync_falsy_strings(self) -> None:
        for falsy in ["0", "false", "no", "off"]:
            result = _coerce_cluster_profile("test", {"scheduler": "slurm", "wandb_sync": falsy})
            assert result.wandb_sync is False

    def test_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            _coerce_cluster_profile("test", ["not a dict"])


class TestLoadClusterCatalogYaml:
    """Test load_cluster_catalog_yaml function."""

    def test_missing_file_returns_default(self) -> None:
        result = load_cluster_catalog_yaml("/nonexistent/catalog.yaml")
        assert "juwels" in result

    def test_empty_yaml_returns_default(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            try:
                result = load_cluster_catalog_yaml(f.name)
                assert "juwels" in result
            finally:
                Path(f.name).unlink()

    def test_valid_catalog(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
clusters:
  test-cluster:
    scheduler: slurm
    cluster_target: test-target
    ssh_target: user@test
""")
            f.flush()
            try:
                result = load_cluster_catalog_yaml(f.name)
                assert "test-cluster" in result
                assert result["test-cluster"].scheduler == "slurm"
            finally:
                Path(f.name).unlink()

    def test_no_clusters_raises(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("clusters: {}")
            f.flush()
            try:
                with pytest.raises(ValueError, match="no cluster entries"):
                    load_cluster_catalog_yaml(f.name)
            finally:
                Path(f.name).unlink()


class TestConfigureActiveCluster:
    """Test configure_active_cluster function."""

    def test_missing_file_loads_default(self) -> None:
        profile = configure_active_cluster("/nonexistent/catalog.yaml")
        assert profile.name == "juwels"

    def test_loads_named_cluster(self, tmp_path: Path) -> None:
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.write_text("""
clusters:
  test-cluster:
    scheduler: slurm
    cluster_target: test-target
    ssh_target: user@test
""")
        profile = configure_active_cluster(str(catalog_path), cluster_name="test-cluster")
        assert profile.name == "test-cluster"

    def test_unknown_cluster_raises(self, tmp_path: Path) -> None:
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.write_text("""
clusters:
  test-cluster:
    scheduler: slurm
""")
        with pytest.raises(ValueError, match="Unknown cluster"):
            configure_active_cluster(str(catalog_path), cluster_name="unknown")


class TestGetActiveClusterProfile:
    """Test get_active_cluster_profile function."""

    def test_returns_valid_profile(self) -> None:
        profile = get_active_cluster_profile()
        assert profile.name in ("juwels", "test-cluster", "custom")
        assert profile.scheduler in ("slurm", "lsf")

    def test_after_configure(self, tmp_path: Path) -> None:
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.write_text("""
clusters:
  custom:
    scheduler: lsf
    cluster_target: custom-target
    ssh_target: user@custom
""")
        configure_active_cluster(str(catalog_path), cluster_name="custom")
        profile = get_active_cluster_profile()
        assert profile.name == "custom"
        assert profile.scheduler == "lsf"


class TestWrapRemoteLoginShell:
    """Test _wrap_remote_login_shell function."""

    def test_wraps_command(self) -> None:
        result = _wrap_remote_login_shell("ls /tmp")
        assert "bash -lc" in result
        assert "ls /tmp" in result


class TestParseSlurmDurationToSeconds:
    """Test _parse_slurm_duration_to_seconds function."""

    def test_standard_format(self) -> None:
        assert _parse_slurm_duration_to_seconds("01:30:45") == 5445

    def test_with_days(self) -> None:
        assert _parse_slurm_duration_to_seconds("2-12:34:56") == 218096

    def test_na_returns_none(self) -> None:
        assert _parse_slurm_duration_to_seconds("N/A") is None

    def test_unknown_returns_none(self) -> None:
        assert _parse_slurm_duration_to_seconds("Unknown") is None

    def test_empty_returns_none(self) -> None:
        assert _parse_slurm_duration_to_seconds("") is None

    def test_invalid_returns_none(self) -> None:
        assert _parse_slurm_duration_to_seconds("invalid") is None

    def test_two_part_format(self) -> None:
        result = _parse_slurm_duration_to_seconds("30:45")
        assert result == 1845


class TestParseLsfDurationToSeconds:
    """Test _parse_lsf_duration_to_seconds function."""

    def test_digit_only(self) -> None:
        assert _parse_lsf_duration_to_seconds("3600") == 3600

    def test_hms_format(self) -> None:
        result = _parse_lsf_duration_to_seconds("01:30:00")
        assert result == 5400

    def test_na_returns_none(self) -> None:
        assert _parse_lsf_duration_to_seconds("N/A") is None

    def test_unknown_returns_none(self) -> None:
        assert _parse_lsf_duration_to_seconds("Unknown") is None

    def test_dash_returns_none(self) -> None:
        assert _parse_lsf_duration_to_seconds("-") is None

    def test_duration_string_minutes(self) -> None:
        result = _parse_lsf_duration_to_seconds("30 minutes")
        assert result == 1800

    def test_duration_string_hours(self) -> None:
        result = _parse_lsf_duration_to_seconds("2 hours")
        assert result == 7200

    def test_duration_string_days(self) -> None:
        result = _parse_lsf_duration_to_seconds("1 day")
        assert result == 86400


class TestSubmitClusterJob:
    """Test submit_cluster_job function."""

    def test_empty_command_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            submit_cluster_job("", "01:00:00")

    def test_invalid_runtime_raises(self) -> None:
        with pytest.raises(ValueError, match="HH:MM:SS"):
            submit_cluster_job("echo test", "1:00:00")


class TestLoadRemoteLog:
    """Test load_remote_log function."""

    def test_invalid_log_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid log id"):
            load_remote_log("invalid log id!", "user@host")

    def test_invalid_log_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid log type"):
            load_remote_log("valid_log_id", log="DEBUG")


class TestGetSlurmJobStatus:
    """Test get_slurm_job_status function."""

    def test_empty_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            get_slurm_job_status("")

    def test_invalid_job_id_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid job id"):
            get_slurm_job_status("job with spaces")


class TestGetLsfJobStatus:
    """Test get_lsf_job_status function."""

    def test_empty_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            get_lsf_job_status("")

    def test_invalid_job_id_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid job id"):
            get_lsf_job_status("job with spaces")


class TestGetClusterJobStatus:
    """Test get_cluster_job_status function."""

    def test_unsupported_scheduler_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            get_cluster_job_status("12345", scheduler="pbs")


class TestGetSlurmJobRuntime:
    """Test get_slurm_job_runtime function."""

    def test_empty_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            get_slurm_job_runtime("")

    def test_invalid_job_id_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid job id"):
            get_slurm_job_runtime("job with spaces")


class TestGetLsfJobRuntime:
    """Test get_lsf_job_runtime function."""

    def test_empty_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            get_lsf_job_runtime("")

    def test_invalid_job_id_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid job id"):
            get_lsf_job_runtime("job with spaces")


class TestGetClusterJobRuntime:
    """Test get_cluster_job_runtime function."""

    def test_unsupported_scheduler_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            get_cluster_job_runtime("12345", scheduler="pbs")