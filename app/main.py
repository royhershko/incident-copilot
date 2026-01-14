from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()


@app.callback()
def main() -> None:
    """Incident Copilot CLI."""
    return


class Alert(BaseModel):
    # Flexible alert schema: supports both "flat" payloads and Prometheus-style labels/annotations.
    alertname: str | None = None
    severity: str | None = None
    summary: str | None = None
    description: str | None = None
    service: str | None = None
    namespace: str | None = None
    labels: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def from_json(data: dict[str, Any]) -> Alert:
        labels = data.get("labels") or {}
        ann = data.get("annotations") or {}

        # Some webhook payloads wrap alerts under {"alerts":[{...}]}
        if "alerts" in data and isinstance(data["alerts"], list) and data["alerts"]:
            first = data["alerts"][0]
            labels = first.get("labels") or labels
            ann = first.get("annotations") or ann
            data = {**data, **first}

        alertname = (
            data.get("alertname")
            or labels.get("alertname")
            or labels.get("alert")
            or labels.get("rule")
        )
        severity = data.get("severity") or labels.get("severity")
        summary = ann.get("summary") or data.get("summary")
        description = ann.get("description") or data.get("description")
        service = labels.get("service") or labels.get("app") or data.get("service")
        namespace = labels.get("namespace") or data.get("namespace")

        return Alert(
            alertname=alertname,
            severity=severity,
            summary=summary,
            description=description,
            service=service,
            namespace=namespace,
            labels=labels,
            annotations=ann,
            raw=data,
        )


@dataclass
class RunbookHit:
    path: Path
    score: float
    title: str
    matched_terms: list[str]


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9_\-:/\. ]+", " ", text)
    return [t for t in text.split() if len(t) > 2]


def load_runbooks(runbooks_dir: Path) -> list[tuple[Path, str]]:
    if not runbooks_dir.exists():
        return []
    files = sorted(runbooks_dir.glob("*.md"))
    runbooks: list[tuple[Path, str]] = []
    for f in files:
        try:
            runbooks.append((f, f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return runbooks


def score_runbook(content: str, terms: list[str]) -> tuple[float, list[str], str]:
    lower = content.lower()

    title = "Untitled"
    for line in content.splitlines():
        if line.strip().startswith("#"):
            title = line.strip().lstrip("#").strip()
            break

    matched = [t for t in terms if t in lower]

    # Simple scoring: match ratio + bonus for title matches
    title_lower = title.lower()
    score = 0.0
    if terms:
        score += len(matched) / max(len(terms), 1)
    score += 0.5 * sum(1 for t in terms if t in title_lower)

    return score, matched[:12], title


def pick_top_runbooks(alert: Alert, runbooks_dir: Path, top_k: int = 3) -> list[RunbookHit]:
    parts = [
        alert.alertname or "",
        alert.summary or "",
        alert.description or "",
        alert.service or "",
        alert.namespace or "",
        json.dumps(alert.labels, ensure_ascii=False),
        json.dumps(alert.annotations, ensure_ascii=False),
    ]
    terms = tokenize(" ".join(parts))

    runbooks = load_runbooks(runbooks_dir)
    hits: list[RunbookHit] = []
    for path, content in runbooks:
        score, matched, title = score_runbook(content, terms)
        if score > 0:
            hits.append(RunbookHit(path=path, score=score, title=title, matched_terms=matched))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def build_triage(alert: Alert, hits: list[RunbookHit]) -> dict[str, Any]:
    service = alert.service or alert.labels.get("service") or "unknown-service"
    ns = alert.namespace or alert.labels.get("namespace") or "default"
    name = alert.alertname or "unknown-alert"

    steps = [
        f"Check scope: is it only {service} in {ns}, or are dependent services impacted too? (dashboards/service map)",
        "Check recent changes: deploy/config/feature flags within the last 30–60 minutes",
        f"Kubernetes quick check: kubectl -n {ns} get pods,events && kubectl -n {ns} describe pod <pod>",
        "Logs: filter by correlation/request-id (if available) and scan for ERROR/timeout spikes",
        "Performance: latency p95/p99, error rate, saturation signals (CPU/Mem/DB pool/queue lag)",
    ]

    text = " ".join([alert.summary or "", alert.description or "", name]).lower()
    hypotheses: list[str] = []

    if any(k in text for k in ["timeout", "timed out", "deadline", "context deadline"]):
        hypotheses.append(
            "Timeouts: dependency degradation (DB/Redis/3rd party) or saturation (threads/CPU/conn pool)."
        )
    if any(k in text for k in ["5xx", "error rate", "http 500", "internal server error"]):
        hypotheses.append(
            "5xx spike: bad deploy, config regression, or dependency returning errors."
        )
    if any(k in text for k in ["oom", "out of memory", "crashloop", "crash loop"]):
        hypotheses.append("Crash/OOM: memory leak, low limits, or unexpected traffic burst.")
    if not hypotheses:
        hypotheses.append(
            "Generic: correlate with recent changes, load increase, dependency health, and networking/DNS."
        )

    return {
        "alert": {
            "name": name,
            "severity": alert.severity,
            "service": service,
            "namespace": ns,
            "summary": alert.summary,
            "description": alert.description,
        },
        "hypotheses": hypotheses[:3],
        "first_steps": steps,
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


def render(triage: dict[str, Any]) -> None:
    a = triage["alert"]
    header = (
        f"[bold]{a['name']}[/bold]  |  severity={a.get('severity')}  |  "
        f"service={a['service']}  |  ns={a['namespace']}"
    )
    console.print(Panel(header, title="Incident Copilot (MVP)", expand=False))

    if a.get("summary"):
        console.print(Panel(a["summary"], title="Summary", expand=False))
    if a.get("description"):
        console.print(Panel(a["description"], title="Description", expand=False))

    t = Table(title="Top Hypotheses", show_header=True, header_style="bold")
    t.add_column("#", style="dim", width=3)
    t.add_column("Hypothesis")
    for i, h in enumerate(triage["hypotheses"], start=1):
        t.add_row(str(i), h)
    console.print(t)

    s = Table(title="First 5 Steps", show_header=True, header_style="bold")
    s.add_column("#", style="dim", width=3)
    s.add_column("Action")
    for i, step in enumerate(triage["first_steps"], start=1):
        s.add_row(str(i), step)
    console.print(s)

    rb = triage["runbooks"]
    if rb:
        rbt = Table(title="Matched Runbooks", show_header=True, header_style="bold")
        rbt.add_column("Score", width=7)
        rbt.add_column("Title")
        rbt.add_column("Path")
        for h in rb:
            rbt.add_row(str(h["score"]), h["title"], h["path"])
        console.print(rbt)
    else:
        console.print("[yellow]No runbooks matched yet. Add docs/runbooks/*.md[/yellow]")


@app.command()
def triage(
    alert_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to alert.json"),
    runbooks_dir: Path = typer.Option(Path("docs/runbooks"), help="Runbooks directory (markdown)"),
    out_json: Path | None = typer.Option(None, help="Optional output JSON file"),
):
    data = json.loads(alert_file.read_text(encoding="utf-8"))
    alert = Alert.from_json(data)
    hits = pick_top_runbooks(alert, runbooks_dir)
    triage_obj = build_triage(alert, hits)

    render(triage_obj)

    if out_json:
        out_json.write_text(json.dumps(triage_obj, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]Saved:[/green] {out_json}")


if __name__ == "__main__":
    app()
