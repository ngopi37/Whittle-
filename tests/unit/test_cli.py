import json

import pytest

from apps.cli import main as cli
from schemas.hardware import CpuProfile, GpuProfile, HardwareProfile, MemoryProfile, StorageProfile


@pytest.fixture(autouse=True)
def _stub_hardware(monkeypatch) -> None:
    profile = HardwareProfile(
        os="Windows",
        architecture="x86_64",
        cpu=CpuProfile(vendor="Intel", model="i7", physical_cores=6, logical_cores=12),
        memory=MemoryProfile(total_gb=16, available_gb=12),
        gpu=GpuProfile(),
        storage=StorageProfile(capacity_gb=500, free_gb=100),
    )

    class _StubProfiler:
        def profile(self) -> HardwareProfile:
            return profile

    monkeypatch.setattr(cli, "HardwareProfiler", _StubProfiler)


def test_hardware_command(capsys) -> None:
    assert cli.main(["hardware"]) == 0
    assert json.loads(capsys.readouterr().out)["os"] == "Windows"


def test_edition_command_defaults_to_free(capsys, monkeypatch) -> None:
    monkeypatch.delenv("SG2_EDITION", raising=False)
    assert cli.main(["edition"]) == 0
    assert json.loads(capsys.readouterr().out)["edition"] == "free"


def test_recommend_with_training_flags(capsys) -> None:
    assert cli.main(["recommend", "--batch-size", "8", "--mixed-precision"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["model_size"] for item in payload] == ["20M", "50M", "100M"]


def test_project_create_show_refresh(tmp_path, capsys) -> None:
    path = tmp_path / "p.json"
    assert cli.main(["project", "create", "demo", "--model", "20M", "--output", str(path)]) == 0
    capsys.readouterr()
    assert cli.main(["project", "show", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "demo"
    assert cli.main(["project", "refresh", str(path)]) == 0
    capsys.readouterr()
    from core.project import ProjectConfig

    assert ProjectConfig.load(path).recommendation is not None


def test_project_show_missing_file_is_friendly(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["project", "show", "does-not-exist.json"])
    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().err


def test_large_model_is_gated_without_pro(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("SG2_EDITION", raising=False)
    code = cli.main(
        ["project", "create", "big", "--model", "200M", "--output", str(tmp_path / "b.json")]
    )
    assert code == 1
    assert "pro edition" in capsys.readouterr().err


def test_pipeline_plan_and_run(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "p.json"
    cli.main(["project", "create", "demo", "--model", "20M", "--output", str(path)])
    capsys.readouterr()
    assert cli.main(["pipeline", "plan", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["model_size"] == "20M"

    source = tmp_path / "s.txt"
    source.write_text("a\nb\n", encoding="utf-8")
    assert (
        cli.main(["pipeline", "run", str(path), "--stage", "data-ingest", "--input", str(source)])
        == 0
    )
    assert "succeeded" in capsys.readouterr().out


def test_pipeline_run_missing_prerequisite_reports_cleanly(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "p.json"
    cli.main(["project", "create", "demo", "--model", "20M", "--output", str(path)])
    capsys.readouterr()
    assert cli.main(["pipeline", "run", str(path), "--stage", "evaluate"]) == 1
    assert "run pretrain first" in capsys.readouterr().err
