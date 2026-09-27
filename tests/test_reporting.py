import json
import runpy
import urllib.request
from dataclasses import asdict
from pathlib import Path

import pytest

from speaker_benchmark.schema import Prediction, Segment
from speaker_benchmark.scoring import score_record

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
compare = runpy.run_path(str(SCRIPTS / "compare_runs.py"))["compare"]


def make_run(root, name, errors=False):
    path = root / name
    path.mkdir()
    model = path / name
    model.mkdir()
    record = {"id": "meeting-0", "duration": 2, "source": {"recording": "meeting"}}
    reference = [Segment("a", "hello there", 0, 1)]
    record["reference"] = record["reference_activity"] = [asdict(s) for s in reference]
    prediction = Prediction([Segment("b", "hello" if errors else "hello there", 0, 1)])
    row = {
        "id": record["id"],
        "status": "ok",
        "wall_seconds": 1,
        "rtf": 0.5,
        "prediction": prediction.to_dict(),
        "scores": score_record(
            {**record, "reference": reference, "reference_activity": reference}, prediction
        ),
    }
    (path / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    (path / "plan.json").write_text(
        json.dumps(
            {
                "manifest_sha256": "shared-manifest",
                "protocol": {"der_collar": 0, "tcp_collar": 5},
                "models": [{"id": name}],
            }
        )
    )
    (model / "scores.json").write_text(json.dumps([row]))
    (model / "predictions").mkdir()
    (model / "predictions" / f"{row['id']}.json").write_text(
        json.dumps({k: v for k, v in row.items() if k != "scores"})
    )
    (model / "worker.json").write_text(
        json.dumps(
            {
                "exit_code": 0,
                "sampled_process_tree_peak_rss_bytes": 123,
            }
        )
    )
    return path


def test_comparison_combines_paired_runs_and_rejects_mismatches(tmp_path):
    a, b = make_run(tmp_path, "a"), make_run(tmp_path, "b", errors=True)
    result = compare([a, b])
    assert [s["cpwer"] for s in result["summary"]] == [0, 0.5]
    assert len(result["meetings"]) == 2
    assert result["audio_seconds"] == 2
    plan = json.loads((b / "plan.json").read_text())
    plan["manifest_sha256"] = "different-audio"
    (b / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="same input manifest"):
        compare([a, b])


def test_comparison_refuses_incomplete_runs(tmp_path):
    a = make_run(tmp_path, "a")
    worker = a / "a" / "worker.json"
    worker.write_text(json.dumps({"exit_code": 1}))
    with pytest.raises(ValueError, match="incomplete run"):
        compare([a])


def test_download_redirect_does_not_forward_credentials_to_cdn():
    handler = runpy.run_path(str(SCRIPTS / "fetch_gated_snapshot.py"))["DownloadRedirect"]()
    request = urllib.request.Request(
        "https://huggingface.co/model",
        headers={
            "Authorization": "Bearer synthetic-test-value",
        },
    )
    redirected = handler.redirect_request(
        request, None, 302, "Found", {}, "https://cdn.example/model"
    )
    assert redirected.get_header("Authorization") is None


def test_comparison_requires_explicit_selection_of_rerun(tmp_path):
    old = make_run(tmp_path, "a")
    rerun_parent = tmp_path / "rerun"
    rerun_parent.mkdir()
    new = make_run(rerun_parent, "a", errors=True)
    with pytest.raises(ValueError, match="Duplicate system"):
        compare([old, new])
    result = compare([old, new], {"a": new})
    assert len(result["summary"]) == 1
    assert result["summary"][0]["cpwer"] == 0.5
    assert result["summary"][0]["source_run"] == str(new)


def test_saved_score_verifier_detects_changed_scores_and_missing_records(tmp_path):
    verify = runpy.run_path(str(SCRIPTS / "verify_scores.py"))["verify"]
    path = make_run(tmp_path, "model")
    result = verify(path)
    assert result["checked_records"] == 1 and not result["score_mismatches"]
    scores = path / "model/scores.json"
    rows = json.loads(scores.read_text())
    rows[0]["scores"]["cpwer"]["errors"] += 1
    scores.write_text(json.dumps(rows))
    assert verify(path)["score_mismatches"] == [{"model": "model", "recording": "meeting-0"}]
    scores.write_text("[]")
    with pytest.raises(ValueError, match="every recording exactly once"):
        verify(path)


def test_streaming_reparse_preserves_raw_run_and_measurements(tmp_path):
    reparse = runpy.run_path(str(SCRIPTS / "reparse_streaming_run.py"))["reparse"]
    verify = runpy.run_path(str(SCRIPTS / "verify_scores.py"))["verify"]
    source = make_run(tmp_path, "stream")
    plan_path = source / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["models"][0]["adapter"] = "vibevoice_streaming"
    plan_path.write_text(json.dumps(plan))
    path = source / "stream/predictions/meeting-0.json"
    row = json.loads(path.read_text())
    row["prediction"] = Prediction(
        [Segment("unassigned", "Environmental"), Segment("1", "hello there")],
        timing="unavailable",
        events=[
            {
                "text": "[Environmental ",
                "emitted_seconds": 0.1,
                "chunk_end_seconds": 0,
                "has_text": True,
            },
            {
                "text": "Sounds]\n Speaker 1:hello there",
                "emitted_seconds": 0.8,
                "chunk_end_seconds": 0.5,
                "has_text": True,
            },
        ],
        metadata={
            "raw_output": "[Environmental Sounds]\n Speaker 1:hello there",
            "realtime_paced": True,
        },
    ).to_dict()
    path.write_text(json.dumps(row))
    original = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "corrected"
    audit = reparse(source, output)
    assert audit[0]["speaker_text_changed"]
    assert audit[0]["speech_event_flags_changed"] == 1
    corrected = json.loads((output / path.relative_to(source)).read_text())
    assert corrected["wall_seconds"] == row["wall_seconds"]
    assert (
        corrected["prediction"]["metadata"]["raw_output"]
        == row["prediction"]["metadata"]["raw_output"]
    )
    for before, after in zip(row["prediction"]["events"], corrected["prediction"]["events"]):
        assert {k: v for k, v in before.items() if k != "has_text"} == {
            k: v for k, v in after.items() if k != "has_text"
        }
    assert not verify(output)["score_mismatches"]
    summary = json.loads((output / "summary.json").read_text())[0]
    assert summary["cpwer"] == 0
    assert summary["mean_first_text_seconds"] == 0.8
    assert original == {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError, match="new and outside"):
        reparse(source, source / "nested")
