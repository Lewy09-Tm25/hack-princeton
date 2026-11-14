"""Tests for E11 Record: workspace layout, contract round-trips, redaction, Parquet, and threads."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from engine.contracts import (
    ControlResult,
    Finding,
    ModelMap,
    Plan,
    ResultRecord,
    Series,
)
from engine.record import Record

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _make_sample_plan(model_id: str = "hf:Qwen/Qwen2.5-0.5B-Instruct@main") -> Plan:
    return Plan(
        check="sycophancy",
        model_id=model_id,
        datasets=["syco/are_you_sure", "syco/pairs"],
        layers=[0, 1],
        controls=["shuffled_labels", "random_direction", "held_out"],
        shift_tests=["paraphrase", "multilingual", "multi_turn"],
        user_context={"domain": "general"},
        estimate_seconds=18.0,
        coverage="Layers 0 to 1 on synthetic placeholder items.",
        not_covered="Real model weights not tested.",
        maturity="established",
        papers=["Sharma et al. 2023", "Rimsky et al. 2024"],
        approved=False,
        is_placeholder=True,
    )


def test_workspace_layout_and_contract_round_trips(tmp_path: Path) -> None:
    rec = Record(tmp_path / "workspace")
    try:
        assert (rec.workspace / "record.sqlite").exists()
        assert (rec.workspace / "models").is_dir()
        assert (rec.workspace / "runs").is_dir()
        assert (rec.workspace / "reports").is_dir()
        assert (rec.workspace / "exports").is_dir()

        # ModelMap round-trip
        mmap = ModelMap.model_validate_json(
            (FIXTURES_DIR / "modelmap_qwen25_05b.json").read_text(encoding="utf-8")
        )
        rec.save_model_map(mmap)
        loaded_mmap = rec.get_model_map(mmap.model_id)
        assert loaded_mmap == mmap
        assert rec.list_model_maps() == [mmap]
        assert rec.get_model_map("unknown_model") is None

        # Series round-trip
        series = Series.model_validate_json(
            (FIXTURES_DIR / "series_pythia70m.json").read_text(encoding="utf-8")
        )
        rec.save_series(series)
        assert rec.get_series(series.series_id) == series
        assert rec.list_series() == [series]
        assert rec.get_series("unknown_series") is None

        # Experiment + Plan versioning and approval
        exp_id = rec.create_experiment(
            question="Did the fine-tune increase sycophancy?",
            model_id=mmap.model_id,
            series_id=None,
        )
        exp = rec.get_experiment(exp_id)
        assert exp is not None
        assert exp["question"] == "Did the fine-tune increase sycophancy?"

        plan = _make_sample_plan(mmap.model_id)
        plan_id = rec.save_plan(exp_id, plan)
        assert plan_id.startswith("plan_")

        saved_v1 = rec.get_plan(plan_id)
        assert saved_v1 is not None
        assert saved_v1.version == 1
        assert saved_v1.approved is False
        assert saved_v1.experiment_id == exp_id

        approved_v1 = rec.approve_plan(plan_id)
        assert approved_v1.approved is True
        assert rec.get_plan(plan_id).approved is True  # type: ignore[union-attr]

        # Revising increments version and resets approved=False
        revised = approved_v1.model_copy(update={"layers": [0, 1, 2]})
        same_plan_id = rec.save_plan(exp_id, revised)
        assert same_plan_id == plan_id
        saved_v2 = rec.get_plan(plan_id)
        assert saved_v2 is not None
        assert saved_v2.version == 2
        assert saved_v2.approved is False
        assert saved_v2.layers == [0, 1, 2]
        assert rec.get_plan(plan_id, version=1).approved is True  # type: ignore[union-attr]

        # Approve v2 and create a run
        rec.approve_plan(plan_id)
        run_id = rec.create_run(plan_id, metadata={"seed": 0, "device": "cpu"})
        rdir = rec.run_dir(run_id)
        assert (rdir / "plan.json").is_file()
        assert (rdir / "code").is_dir()
        assert (rdir / "activations").is_dir()
        assert (rdir / "logs.txt").is_file()

        plan_on_disk = Plan.model_validate_json((rdir / "plan.json").read_text(encoding="utf-8"))
        assert plan_on_disk.plan_id == plan_id
        assert plan_on_disk.version == 2
        assert plan_on_disk.approved is True

        rec.set_run_status(run_id, "running", "Collecting activations")
        rec.set_run_status(run_id, "completed", "Finished synthetic check")
        run_info = rec.get_run(run_id)
        assert run_info is not None
        assert run_info["status"] == "completed"
        assert run_info["message"] == "Finished synthetic check"
        assert run_info["metadata"]["is_placeholder"] is True
        assert len(rec.list_runs()) == 1

        # Add 100 result rows across checkpoints and metrics, verify SQLite & Parquet
        rows: list[ResultRecord] = []
        checkpoints = ["step1000", "step8000", "step16000", "step32000"]
        metrics = ["probe_accuracy", "probe_auroc"]
        for idx in range(100):
            ckpt = checkpoints[idx % len(checkpoints)]
            metric = metrics[idx % len(metrics)]
            rows.append(
                ResultRecord(
                    run_id=run_id,
                    check="sycophancy",
                    component_id=f"L{idx % 6}",
                    checkpoint_id=ckpt,
                    metric=metric,
                    value=0.5 + (idx % 40) * 0.01,
                    control=ControlResult(kind="shuffled_labels", value=0.51, passed=True),
                    dataset_id="syco/pairs_placeholder",
                    split="held_out",
                    seed=0,
                    device="cpu",
                    precision="fp32",
                    quantized=False,
                    is_placeholder=True,
                )
            )
        rec.add_results(rows)

        all_queried = rec.query_results(run_id=run_id)
        assert len(all_queried) == 100

        filtered = rec.query_results(
            run_id=run_id,
            check="sycophancy",
            metric="probe_accuracy",
            checkpoint_id="step1000",
        )
        assert len(filtered) == 25
        assert all(r.metric == "probe_accuracy" and r.checkpoint_id == "step1000" for r in filtered)

        parquet_path = rdir / "results.parquet"
        assert parquet_path.is_file()
        table = pq.read_table(parquet_path)
        assert table.num_rows == 100
        assert set(table.column_names) >= {
            "run_id",
            "check",
            "component_id",
            "checkpoint_id",
            "metric",
            "value",
            "is_placeholder",
        }

        # Finding round-trip (placeholder -> verified=False)
        finding = Finding(
            finding_id="finding_001",
            run_id=run_id,
            check="sycophancy",
            claim="Placeholder probe accuracy separated contrast pairs in layers 2 to 5.",
            strength="suggestive",
            maturity="established",
            evidence=[f"{run_id}:probe_accuracy"],
            controls=[ControlResult(kind="shuffled_labels", value=0.51, passed=True)],
            shift_tests=[
                ControlResult(kind="paraphrase", value=0.76, passed=True),
                ControlResult(kind="multilingual", value=None, passed=None, notes="not run"),
            ],
            coverage="Layers 0 to 5 on synthetic placeholder data, seed 0.",
            not_shown="Placeholder test run; does not measure real model weights.",
            verified=False,
            is_placeholder=True,
        )
        rec.save_finding(finding)
        assert rec.get_finding("finding_001") == finding
        assert rec.get_findings(run_id) == [finding]

        # Reports registration and lookup
        report_file = rec.workspace / "reports" / "report_001.html"
        report_file.write_text("<html><body>test</body></html>", encoding="utf-8")
        rec.register_report("report_001", run_id, report_file)
        assert rec.get_report_path("report_001") == report_file.resolve()
        reports = rec.list_reports()
        assert len(reports) == 1
        assert reports[0]["report_id"] == "report_001"
        assert reports[0]["run_id"] == run_id
    finally:
        rec.close()


def test_examples_redaction_never_exposes_raw_restricted_text(tmp_path: Path) -> None:
    rec = Record(tmp_path / "workspace")
    try:
        exp_id = rec.create_experiment("Refusal check test", "local:tiny_qwen2")
        plan_id = rec.save_plan(exp_id, _make_sample_plan("local:tiny_qwen2"))
        run_id = rec.create_run(plan_id)

        sentinel_prompt_0 = "SYNTHETIC_RESTRICTED_PROMPT_ZERO_DO_NOT_EXPOSE"
        sentinel_prompt_1 = "SYNTHETIC_RESTRICTED_PROMPT_ONE_DO_NOT_EXPOSE"
        sentinel_reply_complied = "alpha beta gamma delta epsilon zeta"
        sentinel_extra_field = "UNEXPECTED_FREE_TEXT_MUST_BE_STRIPPED"

        rec.save_examples(
            run_id,
            [
                {
                    "dataset_id": "refusal/harmful_placeholder",
                    "index": 12,
                    "prompt": sentinel_prompt_0,
                    "response": "I'm sorry, I cannot assist with that request.",
                    "extra_raw_notes": sentinel_extra_field,
                },
                {
                    "dataset_id": "refusal/harmful_placeholder",
                    "index": 47,
                    "prompt": sentinel_prompt_1,
                    "response": sentinel_reply_complied,
                    "reference_response": "I cannot fulfill this request.",
                },
            ],
            contains_harmful=True,
        )

        returned = rec.get_examples(run_id, n=5)
        assert len(returned) == 2

        serialized = json.dumps(returned)
        assert sentinel_prompt_0 not in serialized
        assert sentinel_prompt_1 not in serialized
        assert sentinel_reply_complied not in serialized
        assert sentinel_extra_field not in serialized

        assert returned[0]["prompt"] == "[harmful instruction: refusal/harmful_placeholder #12]"
        assert returned[0]["response"] == "[response: refused]"
        assert returned[0]["redacted"] is True

        assert returned[1]["prompt"] == "[harmful instruction: refusal/harmful_placeholder #47]"
        assert returned[1]["response"] == "[response: complied, 6 words]"
        assert returned[1]["reference_response"] == "[response: refused]"
        assert returned[1]["redacted"] is True

        # Harmless dataset examples remain unredacted when contains_harmful=False
        run_id_benign = rec.create_run(plan_id)
        rec.save_examples(
            run_id_benign,
            [
                {
                    "dataset_id": "syco/are_you_sure_placeholder",
                    "index": 3,
                    "prompt": "What is 2 + 2?",
                    "response": "4",
                }
            ],
            contains_harmful=False,
        )
        benign = rec.get_examples(run_id_benign, n=3)
        assert len(benign) == 1
        assert benign[0]["prompt"] == "What is 2 + 2?"
        assert benign[0]["response"] == "4"
        assert benign[0]["redacted"] is False
    finally:
        rec.close()


def test_concurrent_writes_from_two_threads(tmp_path: Path) -> None:
    rec = Record(tmp_path / "workspace")
    try:
        exp_id = rec.create_experiment("Concurrency test", "local:tiny_qwen2")
        plan_id = rec.save_plan(exp_id, _make_sample_plan("local:tiny_qwen2"))
        run_a = rec.create_run(plan_id)
        run_b = rec.create_run(plan_id)

        errors: list[BaseException] = []

        def writer(target_run_id: str, thread_idx: int) -> None:
            try:
                for batch in range(10):
                    batch_rows = [
                        ResultRecord(
                            run_id=target_run_id,
                            check="sycophancy",
                            component_id=f"L{layer}",
                            checkpoint_id=None,
                            metric="probe_accuracy",
                            value=0.5 + 0.01 * (batch + layer + thread_idx),
                            control=ControlResult(kind="shuffled_labels", value=0.5, passed=True),
                            dataset_id="syco/pairs_placeholder",
                            split="held_out",
                            seed=thread_idx,
                            device="cpu",
                            precision="fp32",
                            quantized=False,
                            is_placeholder=True,
                        )
                        for layer in range(5)
                    ]
                    rec.add_results(batch_rows)
                    rec.save_finding(
                        Finding(
                            finding_id=f"finding_t{thread_idx}_b{batch}",
                            run_id=target_run_id,
                            check="sycophancy",
                            claim=f"Thread {thread_idx} batch {batch} placeholder claim.",
                            strength="suggestive",
                            maturity="established",
                            evidence=[f"{target_run_id}:probe_accuracy"],
                            controls=[
                                ControlResult(kind="shuffled_labels", value=0.5, passed=True)
                            ],
                            shift_tests=[],
                            coverage="Layers 0 to 4 placeholder.",
                            not_shown="Placeholder test.",
                            verified=False,
                            is_placeholder=True,
                        )
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=writer, args=(run_a, 0))
        t2 = threading.Thread(target=writer, args=(run_b, 1))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []
        assert len(rec.query_results(run_id=run_a)) == 50
        assert len(rec.query_results(run_id=run_b)) == 50
        assert pq.read_table(rec.run_dir(run_a) / "results.parquet").num_rows == 50
        assert pq.read_table(rec.run_dir(run_b) / "results.parquet").num_rows == 50
        assert len(rec.get_findings(run_a)) == 10
        assert len(rec.get_findings(run_b)) == 10
    finally:
        rec.close()


def test_approve_unknown_plan_raises_key_error(tmp_path: Path) -> None:
    rec = Record(tmp_path / "workspace")
    try:
        with pytest.raises(KeyError):
            rec.approve_plan("plan_missing")
    finally:
        rec.close()
