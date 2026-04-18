from pathlib import Path

from django.test import Client, TestCase, override_settings

from .models import Experiment, PipelineResult
from .services import (
    _build_command_for_experiment,
    cleanup_missing_output_records,
    delete_experiment_with_artifacts,
    read_terminal_tail,
    reset_experiment_sequences_if_empty,
    request_stop,
)


class ExperimentCommandBuilderTests(TestCase):
    def test_build_train_command(self):
        experiment = Experiment(
            task_type=Experiment.TASK_TRAIN,
            dataset="Cora",
            model_name="grace",
            learning_rate=0.01,
            hidden_dim=256,
            epochs=200,
            temperature=0.5,
            drop_edge_rate=0.2,
            drop_feature_rate=0.2,
            task_params={},
            extra_cli_args=["--seed", "42"],
        )
        cmd, artifacts = _build_command_for_experiment(experiment, config_path="tmp_cfg.yaml", exp1_path="tmp_exp1.csv")
        self.assertIn("--dataset", cmd)
        self.assertIn("Cora", cmd)
        self.assertIn("--method", cmd)
        self.assertIn("grace", cmd)
        self.assertIn("--exp1_log_csv", cmd)
        self.assertIn("tmp_exp1.csv", cmd)
        self.assertEqual(artifacts["exp1_log_csv"], "tmp_exp1.csv")

    def test_build_full_pipeline_multi_command(self):
        experiment = Experiment(
            task_type=Experiment.TASK_FULL_PIPELINE_MULTI,
            dataset="Multi",
            model_name="pipeline-batch",
            task_params={
                "datasets": ["Cora", "PubMed"],
                "continue_on_error": True,
                "baseline_runs": 2,
                "topk_verify": 2,
                "runs_per_top": 2,
                "force_grid": True,
            },
            extra_cli_args=["--std_weight", "0.6"],
        )
        cmd, artifacts = _build_command_for_experiment(experiment, config_path="", exp1_path="")
        self.assertIn("run_selected_full_pipelines.py", " ".join(cmd))
        self.assertIn("--datasets", cmd)
        self.assertIn("Cora", cmd)
        self.assertIn("PubMed", cmd)
        self.assertIn("--continue_on_error", cmd)
        self.assertIn("--force_grid", cmd)
        self.assertIn("--std_weight", cmd)
        self.assertEqual(len(artifacts.get("result_csvs", [])), 2)


class ExperimentStopTests(TestCase):
    def test_request_stop_pending_sets_cancelled(self):
        experiment = Experiment.objects.create(
            task_type=Experiment.TASK_TRAIN,
            dataset="Cora",
            model_name="grace",
            status=Experiment.STATUS_PENDING,
        )
        action = request_stop(experiment)
        experiment.refresh_from_db()
        self.assertEqual(action, "cancelled")
        self.assertEqual(experiment.status, Experiment.STATUS_CANCELLED)

    def test_request_stop_running_sets_flag(self):
        experiment = Experiment.objects.create(
            task_type=Experiment.TASK_GRID_SEARCH,
            dataset="Cora",
            model_name="ifl-gr",
            status=Experiment.STATUS_RUNNING,
        )
        action = request_stop(experiment)
        experiment.refresh_from_db()
        self.assertEqual(action, "requested")
        self.assertTrue(experiment.cancel_requested)


class MonitorApiTests(TestCase):
    @override_settings(ROOT_URLCONF="gcl_system.urls")
    def test_api_returns_terminal_tail(self):
        temp_log = Path("d:/dissertation/openSourceCode/GRACE/logs/test_tail.log")
        temp_log.parent.mkdir(parents=True, exist_ok=True)
        temp_log.write_text("line1\nline2\nline3\n", encoding="utf-8")
        self.addCleanup(lambda: temp_log.unlink(missing_ok=True))

        experiment = Experiment.objects.create(
            task_type=Experiment.TASK_TOP_VERIFY,
            dataset="Cora",
            model_name="ifl-gr",
            status=Experiment.STATUS_RUNNING,
            stdout_path=str(temp_log),
        )
        c = Client()
        resp = c.get(f"/api/experiments/{experiment.pk}/")
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["task_type"], Experiment.TASK_TOP_VERIFY)
        self.assertIn("terminal_tail", payload)
        self.assertIn("line3", payload["terminal_tail"])

    def test_read_terminal_tail_handles_missing_file(self):
        payload = read_terminal_tail("d:/dissertation/openSourceCode/GRACE/logs/not_exists_foo.log")
        self.assertEqual(payload["terminal_tail"], "")


class ArtifactViewTests(TestCase):
    @override_settings(ROOT_URLCONF="gcl_system.urls")
    def test_artifact_file_serves_log_csv(self):
        temp_csv = Path("d:/dissertation/openSourceCode/GRACE/logs/test_preview.csv")
        temp_csv.parent.mkdir(parents=True, exist_ok=True)
        temp_csv.write_text("epoch,loss\n1,0.5\n", encoding="utf-8")
        self.addCleanup(lambda: temp_csv.unlink(missing_ok=True))

        c = Client()
        resp = c.get("/artifacts/logs/test_preview.csv")
        self.assertEqual(resp.status_code, 200)
        content = b"".join(resp.streaming_content)
        self.assertIn(b"epoch,loss", content)

    @override_settings(ROOT_URLCONF="gcl_system.urls")
    def test_results_page_renders_result_files(self):
        temp_csv = Path("d:/dissertation/openSourceCode/GRACE/results/test_full_pipeline_results.csv")
        temp_csv.parent.mkdir(parents=True, exist_ok=True)
        temp_csv.write_text(
            "timestamp,dataset,stage,method,candidate_rank,run_idx,F1Mi_mean,F1Mi_std,F1Ma_mean,F1Ma_std,robust_score,delta_vs_grace,params_json,notes\n"
            "2026-04-17 10:00:00,Cora,summary,grace,0,1,0.81,0.01,0.80,0.01,0.805,0.0,{},ok\n",
            encoding="utf-8",
        )
        self.addCleanup(lambda: temp_csv.unlink(missing_ok=True))

        c = Client()
        resp = c.get("/results/")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "全流程结果文件")
        self.assertContains(resp, "test_full_pipeline_results.csv")


class CleanupMissingOutputTests(TestCase):
    @override_settings(ROOT_URLCONF="gcl_system.urls")
    def test_history_request_prunes_completed_train_experiment_with_missing_exp1_csv(self):
        missing_csv = Path("d:/dissertation/openSourceCode/GRACE/logs/missing_train_exp1.csv")
        missing_csv.unlink(missing_ok=True)

        experiment = Experiment.objects.create(
            task_type=Experiment.TASK_TRAIN,
            dataset="Cora",
            model_name="grace",
            status=Experiment.STATUS_SUCCEEDED,
            exp1_log_path=str(missing_csv),
            artifacts={"exp1_log_csv": str(missing_csv)},
        )

        c = Client()
        resp = c.get("/history/")
        self.assertEqual(resp.status_code, 200)
        self.assertNotContains(resp, f"/experiments/{experiment.pk}/")
        self.assertFalse(Experiment.objects.filter(pk=experiment.pk).exists())

    def test_cleanup_missing_output_records_prunes_stale_pipeline_rows(self):
        missing_csv = Path("d:/dissertation/openSourceCode/GRACE/results/missing_pipeline.csv")
        missing_csv.unlink(missing_ok=True)

        PipelineResult.objects.create(
            dataset="Cora",
            method_key="grace",
            method_name="GRACE",
            stage=PipelineResult.STAGE_SUMMARY,
            F1Mi_mean=0.81,
            source_csv=str(missing_csv),
        )

        result = cleanup_missing_output_records()
        self.assertEqual(result["deleted_pipeline_rows"], 1)
        self.assertEqual(PipelineResult.objects.count(), 0)


class DeleteExperimentTests(TestCase):
    @override_settings(ROOT_URLCONF="gcl_system.urls")
    def test_delete_view_removes_experiment_and_owned_artifacts(self):
        stdout_log = Path("d:/dissertation/openSourceCode/GRACE/logs/delete_task_stdout.log")
        exp1_csv = Path("d:/dissertation/openSourceCode/GRACE/logs/delete_task_exp1.csv")
        curve_png = Path("d:/dissertation/openSourceCode/GRACE/logs/delete_task_exp1_curves.png")
        for path, content in [
            (stdout_log, "stdout\n"),
            (exp1_csv, "epoch,loss\n1,0.5\n"),
            (curve_png, "png"),
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self.addCleanup(lambda p=path: p.unlink(missing_ok=True))

        experiment = Experiment.objects.create(
            task_type=Experiment.TASK_TRAIN,
            dataset="Cora",
            model_name="grace",
            status=Experiment.STATUS_SUCCEEDED,
            stdout_path=str(stdout_log),
            exp1_log_path=str(exp1_csv),
            artifacts={"stdout_log": str(stdout_log), "exp1_log_csv": str(exp1_csv)},
        )

        c = Client()
        resp = c.post(f"/experiments/{experiment.pk}/delete/", {"next": "/history/"})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.headers["Location"], "/history/")
        self.assertFalse(Experiment.objects.filter(pk=experiment.pk).exists())
        self.assertFalse(stdout_log.exists())
        self.assertFalse(exp1_csv.exists())
        self.assertFalse(curve_png.exists())
        recreated = Experiment.objects.create(
            task_type=Experiment.TASK_TRAIN,
            dataset="Cora",
            model_name="grace",
            status=Experiment.STATUS_PENDING,
        )
        self.assertEqual(recreated.pk, 1)

    def test_delete_service_removes_pipeline_rows_for_deleted_result_csv(self):
        result_csv = Path("d:/dissertation/openSourceCode/GRACE/results/delete_me_full_pipeline_results.csv")
        result_csv.parent.mkdir(parents=True, exist_ok=True)
        result_csv.write_text("dataset,method,stage\nCora,grace,summary\n", encoding="utf-8")
        self.addCleanup(lambda: result_csv.unlink(missing_ok=True))

        experiment = Experiment.objects.create(
            task_type=Experiment.TASK_FULL_PIPELINE_SINGLE,
            dataset="Cora",
            model_name="pipeline",
            status=Experiment.STATUS_SUCCEEDED,
            artifacts={"result_csv": str(result_csv)},
        )
        PipelineResult.objects.create(
            dataset="Cora",
            method_key="grace",
            method_name="GRACE",
            stage=PipelineResult.STAGE_SUMMARY,
            F1Mi_mean=0.81,
            source_csv=str(result_csv),
        )

        result = delete_experiment_with_artifacts(experiment)
        self.assertEqual(result["deleted_experiment_id"], experiment.pk)
        self.assertFalse(result_csv.exists())
        self.assertEqual(PipelineResult.objects.count(), 0)

    def test_delete_service_keeps_shared_result_file(self):
        result_csv = Path("d:/dissertation/openSourceCode/GRACE/results/shared_grid_search_results.csv")
        result_csv.parent.mkdir(parents=True, exist_ok=True)
        result_csv.write_text("dataset,method\nCora,ifl-gr\n", encoding="utf-8")
        self.addCleanup(lambda: result_csv.unlink(missing_ok=True))

        exp_a = Experiment.objects.create(
            task_type=Experiment.TASK_GRID_SEARCH,
            dataset="Cora",
            model_name="ifl-gr",
            status=Experiment.STATUS_SUCCEEDED,
            artifacts={"result_csv": str(result_csv)},
        )
        exp_b = Experiment.objects.create(
            task_type=Experiment.TASK_GRID_SEARCH,
            dataset="Cora",
            model_name="ifl-gr",
            status=Experiment.STATUS_SUCCEEDED,
            artifacts={"result_csv": str(result_csv)},
        )

        result = delete_experiment_with_artifacts(exp_a)
        self.assertFalse(Experiment.objects.filter(pk=exp_a.pk).exists())
        self.assertTrue(Experiment.objects.filter(pk=exp_b.pk).exists())
        self.assertTrue(result_csv.exists())
        self.assertIn(str(result_csv.resolve()), result["skipped_files"])

    def test_delete_service_rejects_running_task(self):
        experiment = Experiment.objects.create(
            task_type=Experiment.TASK_TRAIN,
            dataset="Cora",
            model_name="grace",
            status=Experiment.STATUS_RUNNING,
        )

        with self.assertRaises(ValueError):
            delete_experiment_with_artifacts(experiment)

    def test_reset_sequence_helper_resets_advanced_empty_table(self):
        experiment = Experiment.objects.create(
            task_type=Experiment.TASK_TRAIN,
            dataset="Cora",
            model_name="grace",
            status=Experiment.STATUS_PENDING,
        )
        experiment.delete()

        self.assertFalse(Experiment.objects.exists())
        self.assertTrue(reset_experiment_sequences_if_empty())

        recreated = Experiment.objects.create(
            task_type=Experiment.TASK_TRAIN,
            dataset="Cora",
            model_name="grace",
            status=Experiment.STATUS_PENDING,
        )
        self.assertEqual(recreated.pk, 1)
