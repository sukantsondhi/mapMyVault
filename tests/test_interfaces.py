import tempfile
import unittest
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlencode

from src.actions import ActionManager
from src.answers import deterministic_answer
from src.app_server import AppState, create_handler, require_index, require_loopback_host
from src.config import MapperConfig
from src.mapper import RepositoryMapper
from src.mcp_server import create_server
from src.storage import IndexStore
from src import streamlit_app as studio
from src.terminal_ui import interactive_map_config
from tests.test_mapper import FakeAI
from unittest.mock import patch


class InterfaceTests(unittest.TestCase):
    def test_remote_ollama_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            MapperConfig(ollama_url="https://example.com")

    def test_non_loopback_mcp_binding_is_rejected(self):
        with self.assertRaises(ValueError):
            create_server(Path("."), host="0.0.0.0")

    def test_non_loopback_app_binding_is_rejected(self):
        with self.assertRaises(ValueError):
            require_loopback_host("0.0.0.0")

    def test_app_requires_existing_index_for_chat_and_status(self):
        with tempfile.TemporaryDirectory() as base:
            with self.assertRaises(ValueError):
                require_index(Path(base) / "missing-output")

    def test_app_status_returns_json_error_for_missing_index(self):
        with tempfile.TemporaryDirectory() as base, ThreadingHTTPServer(
            ("127.0.0.1", 0), create_handler(AppState())
        ) as server:
            worker = Thread(target=server.handle_request)
            worker.start()
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                query = urlencode({"output": str(Path(base) / "missing-output")})
                connection.request("GET", f"/api/status?{query}")
                response = connection.getresponse()
                payload = json.loads(response.read())
                self.assertEqual(response.status, 400)
                self.assertIn("index not found", payload["error"])
            finally:
                connection.close()
                worker.join(timeout=5)

    def test_deterministic_count_answer_uses_found_count_and_paths(self):
        answer = deterministic_answer(
            {
                "question": "how many offers?",
                "found_count": 2,
                "paths": ["a.pdf", "b.pdf"],
            }
        )
        self.assertIn("2 matching files", answer)
        self.assertIn("`a.pdf`", answer)
        self.assertIn("`b.pdf`", answer)

    def test_actions_require_approval(self):
        with tempfile.TemporaryDirectory() as base:
            base = Path(base)
            repo = base / "repo"
            output = base / "output"
            repo.mkdir()
            (repo / "old.txt").write_text("content", encoding="utf-8")
            mapper = RepositoryMapper(repo, output, ai=FakeAI(), use_chroma=False)
            mapper.map()
            mapper.close()

            actions = ActionManager(output)
            plan = actions.propose_moves(
                [{"source": "old.txt", "destination": "new.txt"}]
            )
            with self.assertRaises(ValueError):
                actions.apply(plan["id"])
            actions.approve(plan["id"])
            actions.apply(plan["id"])
            self.assertTrue((repo / "new.txt").exists())
            self.assertFalse((repo / "old.txt").exists())
            actions.rollback(plan["id"])
            self.assertTrue((repo / "old.txt").exists())
            self.assertFalse((repo / "new.txt").exists())
            actions.close()

    def test_interactive_setup_uses_local_model_choices(self):
        with tempfile.TemporaryDirectory() as base:
            repo = Path(base) / "repo"
            output = Path(base) / "output"
            repo.mkdir()
            answers = [str(repo), str(output)]
            models = ["llama3.1:8b", "nomic-embed-text:latest"]
            with patch("src.terminal_ui.inquirer.text", side_effect=answers) as text, patch(
                "src.terminal_ui.inquirer.list_input", side_effect=models
            ), patch("src.terminal_ui.inquirer.confirm", side_effect=[False, False, True]), patch(
                "src.terminal_ui.LocalAI.models", return_value=models
            ) as model_list:
                chosen_repo, chosen_output, config = interactive_map_config()

            self.assertEqual(chosen_repo, repo.resolve())
            self.assertEqual(chosen_output, output.resolve())
            self.assertEqual(config.generation_model, "llama3.1:8b")
            self.assertEqual(config.embedding_model, "nomic-embed-text:latest")
            self.assertEqual(model_list.call_count, 1)
            self.assertEqual(text.call_args_list[0].kwargs["default"], "")
            self.assertEqual(text.call_args_list[1].kwargs["default"], "")


class StudioTests(unittest.TestCase):
    def _indexed_app(self):
        from streamlit.testing.v1 import AppTest

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name) / "source"
        output = Path(temporary.name) / "output"
        (source / "notes").mkdir(parents=True)
        (source / "notes" / "project.txt").write_text("Project planning notes.", encoding="utf-8")
        mapper = RepositoryMapper(source, output, MapperConfig(ocr_language="fra"), ai=FakeAI(), use_chroma=False)
        try:
            mapper.map(export_vault_notes=False)
        finally:
            mapper.close()
        app = AppTest.from_file(str(Path(studio.__file__)), default_timeout=20).run()
        app.text_input(key="active_output_path_input").set_value(str(output)).run()
        self.assertEqual(len(app.exception), 0)
        return app, source, output

    def test_loaded_explorer_navigates_and_filters(self):
        with patch("src.local_ai.LocalAI.models", return_value=["llama3.1:8b", "nomic-embed-text:latest"]):
            app, source, output = self._indexed_app()
            app.button(key="nav_Explorer").click().run()
            self.assertEqual(len(app.exception), 0)
            app.selectbox(key=f"open_folder_{output}_").select("notes").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["graph_folder"], "notes")
            app.text_input(key="explorer_filter").set_value("project").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.dataframe[0].value.iloc[0]["Name"], "project.txt")
            app.button(key="graph_back").click().run()
            self.assertEqual(app.session_state["graph_folder"], "")
            self.assertEqual(app.text_input(key="explorer_filter").value, "")
            app.text_input(key="graph_new_folder").set_value("new-folder").run()
            app.button(key="graph_create_folder").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue((source / "new-folder").is_dir())
            self.assertEqual(app.session_state["graph_folder"], "new-folder")

    def test_restoring_exclusions_clears_saved_update_patterns(self):
        with tempfile.TemporaryDirectory() as base:
            output = Path(base)
            studio._set_index_excluded_paths(output, ["private"])
            config = MapperConfig(excludes=["keep/**", "private", "private/", "private/**"])
            (output / "data" / "manifest.json").write_text(
                json.dumps({"config": config.to_dict()}), encoding="utf-8"
            )
            studio._restore_index_paths(output, ["private"])
            self.assertEqual(studio._index_excluded_paths(output), [])
            self.assertEqual(studio._graph_update_config(output).excludes, ["keep/**"])

    def test_index_switch_clears_chat_and_loads_saved_settings(self):
        with patch("src.local_ai.LocalAI.models", return_value=["llama3.1:8b", "nomic-embed-text:latest"]):
            app, _source, output = self._indexed_app()
            app.chat_input[0].set_value("how many zzznonexistent?").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.session_state["chat_history"]), 2)
            self.assertIn("0 matching files", app.session_state["chat_history"][-1]["content"])
            app.button(key="nav_Knowledge").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.text_input(key="ocr_language").value, "fra")
            app.text_input(key="active_output_path_input").set_value(str(output.parent / "other")).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["chat_history"], [])
            self.assertEqual(app.session_state["source_path"], "")

    def test_source_selection_includes_more_than_300_items(self):
        with tempfile.TemporaryDirectory() as base:
            source = Path(base)
            for position in range(301):
                (source / f"item-{position}").mkdir()
            self.assertEqual(len(studio._source_tree_items(source)), 301)

    def test_graph_update_preserves_saved_extraction_settings(self):
        config = MapperConfig(enable_ocr=False, enable_vision=True, ocr_language="fra")
        with patch.object(studio, "_config_from_manifest", return_value=config), patch.object(
            studio, "_index_excluded_paths", return_value=["private"]
        ):
            updated = studio._graph_update_config(Path("output"))
        self.assertFalse(updated.enable_ocr)
        self.assertTrue(updated.enable_vision)
        self.assertEqual(updated.ocr_language, "fra")
        self.assertIn("private/**", updated.excludes)

    def test_zero_count_does_not_call_chat_model(self):
        with patch.object(studio, "VaultIndex") as index_class, patch.object(studio, "LocalAI") as ai:
            index_class.return_value.ask_mapmyvault.return_value = {
                "question": "how many invoices?", "found_count": 0, "paths": []
            }
            result = studio._ask_with_history(Path("output"), "how many invoices?", [], "model")
        self.assertTrue(result["deterministic"])
        self.assertIn("0 matching files", result["answer"])
        ai.assert_not_called()

    def test_studio_views_render_without_an_index(self):
        from streamlit.testing.v1 import AppTest

        with patch("src.local_ai.LocalAI.models", return_value=["llama3.1:8b", "nomic-embed-text:latest"]):
            app = AppTest.from_file(str(Path(studio.__file__)), default_timeout=20).run()
            self.assertEqual(len(app.exception), 0)
            for view in ("Knowledge", "Explorer", "Chat"):
                app.button(key=f"nav_{view}").click().run()
                self.assertEqual(len(app.exception), 0, view)


class ActionSafetyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name) / "source"
        self.repository.mkdir()
        output = Path(temporary.name) / "output"
        store = IndexStore(output / "data" / "index.sqlite")
        store.set_metadata("repository", str(self.repository))
        store.connection.commit()
        store.close()
        self.actions = ActionManager(output)
        self.addCleanup(self.actions.close)
        for name in ("first.txt", "second.txt"):
            (self.repository / name).write_text(name, encoding="utf-8")
        self.operations = [
            {"source": "first.txt", "destination": "moved/first.txt"},
            {"source": "second.txt", "destination": "moved/second.txt"},
        ]

    def _approved_plan(self):
        plan = self.actions.propose_moves(self.operations)
        self.actions.approve(plan["id"])
        return plan["id"]

    def test_apply_does_not_overwrite_new_destination(self):
        plan_id = self._approved_plan()
        destination = self.repository / "moved/second.txt"
        destination.parent.mkdir()
        destination.write_text("new data", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Destination already exists"):
            self.actions.apply(plan_id)
        self.assertEqual(destination.read_text(encoding="utf-8"), "new data")
        self.assertTrue((self.repository / "first.txt").exists())

    def test_apply_checks_entire_batch_before_moving(self):
        plan_id = self._approved_plan()
        (self.repository / "second.txt").unlink()
        with self.assertRaisesRegex(ValueError, "Source does not exist"):
            self.actions.apply(plan_id)
        self.assertTrue((self.repository / "first.txt").exists())

    def test_apply_checks_destination_parents_before_moving(self):
        self.operations[1]["destination"] = "blocked/second.txt"
        plan_id = self._approved_plan()
        (self.repository / "blocked").write_text("new file", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "parent is not a directory"):
            self.actions.apply(plan_id)
        self.assertTrue((self.repository / "first.txt").exists())

    def test_rollback_checks_entire_batch_before_moving(self):
        plan_id = self._approved_plan()
        self.actions.apply(plan_id)
        (self.repository / "first.txt").write_text("new data", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Rollback blocked"):
            self.actions.rollback(plan_id)
        self.assertTrue((self.repository / "moved/second.txt").exists())
        self.assertFalse((self.repository / "second.txt").exists())

    def test_applied_plan_cannot_be_approved_again(self):
        plan_id = self._approved_plan()
        self.actions.apply(plan_id)
        with self.assertRaisesRegex(ValueError, "Only a proposed"):
            self.actions.approve(plan_id)

    def test_overlapping_and_empty_plans_are_invalid(self):
        batches = [
            [],
            [self.operations[0], self.operations[0]],
            [{"source": "first.txt", "destination": "first.txt/child"}],
            [
                {"source": "first.txt", "destination": "new.txt"},
                {"source": "second.txt", "destination": "new.txt"},
            ],
        ]
        for operations in batches:
            with self.subTest(operations=operations):
                plan = self.actions.propose_moves(operations)
                self.assertFalse(plan["validation"]["valid"])


if __name__ == "__main__":
    unittest.main()
