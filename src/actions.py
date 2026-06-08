"""Approval-gated move and rename action plans."""

from pathlib import Path
from typing import Dict, List
import json
import shutil
import uuid

from .storage import IndexStore, utc_now


class ActionManager:
    def __init__(self, output: Path):
        self.output = output.resolve()
        self.store = IndexStore(self.output / "data" / "index.sqlite")
        repository = self.store.get_metadata("repository")
        if not repository:
            manifest = json.loads((self.output / "data" / "manifest.json").read_text())
            repository = manifest["repository"]
        self.repository = Path(repository).resolve()

    def close(self) -> None:
        self.store.close()

    def propose_moves(self, operations: List[Dict[str, str]]) -> Dict:
        errors = []
        normalized = []
        for operation in operations:
            source = (self.repository / operation["source"]).resolve()
            destination = (self.repository / operation["destination"]).resolve()
            if self.repository not in source.parents or self.repository not in destination.parents:
                errors.append("All action paths must remain inside the repository")
            elif not source.exists():
                errors.append(f"Source does not exist: {operation['source']}")
            elif destination.exists():
                errors.append(f"Destination already exists: {operation['destination']}")
            normalized.append(operation)
        plan_id = uuid.uuid4().hex
        validation = {"valid": not errors, "errors": errors}
        self.store.connection.execute(
            "INSERT INTO action_plans VALUES(?,?,?,?,NULL,NULL,?)",
            (plan_id, "proposed", json.dumps(normalized), json.dumps(validation), utc_now()),
        )
        self.store.connection.commit()
        return {"id": plan_id, "status": "proposed", "validation": validation}

    def approve(self, plan_id: str) -> Dict:
        row = self._plan(plan_id)
        validation = json.loads(row["validation_json"])
        if not validation["valid"]:
            raise ValueError("Invalid action plan cannot be approved")
        self.store.connection.execute(
            "UPDATE action_plans SET status='approved',approved_at=? WHERE id=?",
            (utc_now(), plan_id),
        )
        self._audit(plan_id, "approved", {})
        self.store.connection.commit()
        return {"id": plan_id, "status": "approved"}

    def apply(self, plan_id: str) -> Dict:
        row = self._plan(plan_id)
        if row["status"] != "approved":
            raise ValueError("Action plan requires explicit approval")
        applied = []
        for operation in json.loads(row["operations_json"]):
            source = self.repository / operation["source"]
            destination = self.repository / operation["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            applied.append(operation)
        self.store.connection.execute(
            "UPDATE action_plans SET status='applied',applied_at=? WHERE id=?",
            (utc_now(), plan_id),
        )
        self._audit(plan_id, "applied", {"operations": applied})
        self.store.connection.commit()
        return {"id": plan_id, "status": "applied", "operations": applied}

    def rollback(self, plan_id: str) -> Dict:
        row = self._plan(plan_id)
        if row["status"] != "applied":
            raise ValueError("Only an applied action plan can be rolled back")
        reversed_operations = []
        for operation in reversed(json.loads(row["operations_json"])):
            source = self.repository / operation["destination"]
            destination = self.repository / operation["source"]
            if not source.exists() or destination.exists():
                raise ValueError(
                    f"Rollback blocked for {operation['destination']} -> {operation['source']}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            reversed_operations.append(
                {"source": operation["destination"], "destination": operation["source"]}
            )
        self.store.connection.execute(
            "UPDATE action_plans SET status='rolled_back' WHERE id=?", (plan_id,)
        )
        self._audit(plan_id, "rolled_back", {"operations": reversed_operations})
        self.store.connection.commit()
        return {"id": plan_id, "status": "rolled_back", "operations": reversed_operations}

    def _plan(self, plan_id: str):
        row = self.store.connection.execute(
            "SELECT * FROM action_plans WHERE id=?", (plan_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Unknown action plan: {plan_id}")
        return row

    def _audit(self, plan_id: str, event: str, details: Dict) -> None:
        self.store.connection.execute(
            "INSERT INTO action_audit(plan_id,event,details_json,created_at) VALUES(?,?,?,?)",
            (plan_id, event, json.dumps(details), utc_now()),
        )
