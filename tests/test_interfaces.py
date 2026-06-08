import tempfile
import unittest
from pathlib import Path

from src.actions import ActionManager
from src.answers import deterministic_answer
from src.app_server import require_index, require_loopback_host
from src.config import MapperConfig
from src.mapper import RepositoryMapper
from src.mcp_server import create_server
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


if __name__ == "__main__":
    unittest.main()
