from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()


class Alert(BaseModel):
    alertname: str | None = None
    severity: str | None = None
    summary: str | None = None
    description: str | None = None
    service: str | None = None
    namespace: str | None = None
    labels: Dict[str, Any] = Field(default_factory=dict)
    annotations: Dict[str, Any] = Field(default_factory=dict)
    raw: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Alert":
        labels = data.get("labels") or {}
        annotations = data.get("annotations") or {}

        if "alerts" in data and isinstance(data["alerts"], list) and data["alerts"]:
            first = data["alerts"][0]
            labels = first.get("labels") or labels
            annotations = first.get("annotations") or annotations
            data = {**data, **first}

        return Alert(
            alertname=data.get("alertname") or labels.get("alertname"),
            severity=data.get("severity") or labels.get("severity"),
            summary=annotations.get("summary"),
            description=annotations.get("description"),
            service=labels.get("service"),
            namespace=labels.get("namespace"),
            labels=labels,
            annotations=annotations,
            raw=data,
        )


@dataclass
class RunbookHit:
    path: Path
    score: float
    title: str
    matched_terms: List[str]


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_\-:/\. ]+", " ", text)
    return [t for t in text.split() if len(t) > 2]


def load_runbooks(runbooks_dir: Path) -> List[Tuple[Path, str]]:
    if not runbooks_dir.exists():
        return []

    runbooks: List[Tuple[Path, str]] = []
    for file in runbooks_dir.glob("*.md"):
        runbooks.append((file, file.read_text(encoding="utf-8")))
    return runbooks


def score_runbook(content: str, terms: List[str]) -> Tuple[float, List[str], str]:
    lower = content.lower()
    title = "Untitled"

    for line in content.splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            break

    matched = [t for t in terms if t in lower]
    score = len(matched) / max(len(terms), 1)
    score += 0.5 * sum(1 for t in terms if t in title.lower())

    return score, matched[:10], title


def pick_top_runbooks(alert: Alert, runbooks_dir: Path, top_k: int = 3) -> List[RunbookHit]:
    text = " ".join(
        [
            alert.alertname or "",
            alert.summary or "",
            alert.description or "",
            alert.service or "",
            alert.namespace or "",
            json.dumps(alert.labels),
            json.dumps(alert.annotations),
        ]
    )
    terms = tokenize(text)

    hits: List[RunbookHit] = []
    for path, content in load_runbooks(runbooks_dir):
        score, matched, title = score_runbook(content, terms)
        if score > 0:
            hits.append(RunbookHit(path, score, title, matched))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def build_triage(alert: Alert, hits: List[RunbookHit]) -> Dict[str, Any]:
    hypotheses = []
    text = (alert.summary or "").lower() + " " + (alert.description or "").lower()

    if "5xx" in text or "500" in text:
        hypotheses.append("5xx spike: bad deploy, config regression, or failing dependency.")
    if "timeout" in text:
        hypotheses.append("Timeouts: saturation or slow downstream dependency.")

    if not hypotheses:
        hypotheses.append("General investigation: recent change, load, or dependency issue.")

    return {
        "alert": {
            "name": alert.alertname,
            "severity": alert.severity,
            "service": alert.service,
            "namespace": alert.namespace,
            "summary": alert.summary,
            "description": alert.description,
        },
        "hypotheses": hypotheses[:3],
        "first_steps": [
            "Check scope: single service or cascading impact",
            "Check recent deploys/config changes",
            "Inspect pods and events",
            "Scan logs for errors/timeouts",
            "Check latency and saturation metrics",
        ],
        "runbooks": [
            {
                "title": h.title,
                "path": str(h.path),
                "score": round(h.score, 3),
                "matched_terms": h.matched_terms,
            }
            for h in hits
        ],
    }


def render(triage: Dict[str, Any]) -> None:
    alert = triage["alert"]
    console.print(
        Panel(
            f"{alert['name']} | severity={alert['severity']} | "
            f"service={alert['service']} | ns={alert['namespace']}",
            title="Incident Copilot (MVP)",
        )
    )

    if alert["summary"]:
        console.print(Panel(alert["summary"], title="Summary"))
    if alert["description"]:
        console.print(Panel(alert["description"], title="Description"))

    table = Table(title="Top Hypotheses")
    table.add_column("#", width=3)
    table.add_column("Hypothesis")

    for i, h in enumerate(triage["hypotheses"], start=1):
        table.add_row(str(i), h)

    console.print(table)


@app.command()
def triage(
    alert_file: Path = typer.Argument(..., exists=True, readable=True),
    runbooks_dir: str = typer.Option("docs/runbooks", help="Runbooks directory"),
    out_json: Path | None = typer.Option(None, help="Optional output JSON file"),
) -> None:
    runbooks_path = Path(runbooks_dir)
    data = json.loads(alert_file.read_text(encoding="utf-8"))
    alert = Alert.from_json(data)
    hits = pick_top_runbooks(alert, runbooks_path)
    triage_obj = build_triage(alert, hits)

    render(triage_obj)

    if out_json:
        out_json.write_text(json.dumps(triage_obj, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]Saved:[/green] {out_json}")


if __name__ == "__main__":
    app()
