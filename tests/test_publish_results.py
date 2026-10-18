import json

import pytest

from scripts.publish_registered_results import expected_reports, publish


def test_registered_report_publication_is_complete_and_immutable(tmp_path):
    artifacts = tmp_path / "artifacts"
    expected = expected_reports()
    assert len(expected) == 404
    assert len(set(expected)) == len(expected)
    for index, relative in enumerate(expected):
        path = artifacts / "reports" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"registered_report": index}) + "\n", encoding="utf-8")
    output = tmp_path / "published"
    report = publish(artifacts, output)
    assert report["status"] == "all_registered_reports_published"
    assert report["expected_report_count"] == 404
    assert publish(artifacts, output) == report
    first = output / expected[0]
    first.write_text("changed\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        publish(artifacts, output)


def test_registered_report_publication_rejects_missing_report(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing registered reports"):
        publish(tmp_path / "artifacts", tmp_path / "published")
