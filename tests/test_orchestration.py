from pathlib import Path


def test_mechanism_reports_use_configured_artifact_root():
    script = Path("scripts/report_registered.ps1").read_text(encoding="utf-8")
    assert (
        '--artifacts $Artifacts --output (Join-Path $Artifacts "reports/mechanisms")'
        in script
    )
