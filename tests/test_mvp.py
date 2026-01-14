from pathlib import Path

from app.main import Alert, build_triage, pick_top_runbooks


def test_alert_parsing_prom_style():
    payload = {
        "labels": {
            "alertname": "HighErrorRate",
            "severity": "critical",
            "service": "trade-api",
            "namespace": "trading",
        },
        "annotations": {"summary": "5xx spike", "description": "HTTP 500 rate > 5%"},
    }
    a = Alert.from_json(payload)
    assert a.alertname == "HighErrorRate"
    assert a.severity == "critical"
    assert a.service == "trade-api"
    assert a.namespace == "trading"
    assert a.summary == "5xx spike"


def test_runbook_matching(tmp_path: Path):
    runbooks_dir = tmp_path / "runbooks"
    runbooks_dir.mkdir(parents=True, exist_ok=True)

    (runbooks_dir / "k8s-5xx.md").write_text(
        "# K8s High Error Rate\n\nMentions: 5xx spike on trade-api and namespace trading.\n",
        encoding="utf-8",
    )

    payload = {
        "labels": {
            "alertname": "HighErrorRate",
            "severity": "critical",
            "service": "trade-api",
            "namespace": "trading",
        },
        "annotations": {
            "summary": "5xx spike on trade-api",
            "description": "HTTP 500 rate > 5% for 10m",
        },
    }
    a = Alert.from_json(payload)
    hits = pick_top_runbooks(a, runbooks_dir, top_k=3)
    assert len(hits) >= 1
    assert hits[0].path.name == "k8s-5xx.md"
    assert hits[0].score > 0


def test_build_triage_shape():
    payload = {
        "labels": {
            "alertname": "HighErrorRate",
            "severity": "critical",
            "service": "trade-api",
            "namespace": "trading",
        },
        "annotations": {
            "summary": "5xx spike on trade-api",
            "description": "HTTP 500 rate > 5% for 10m",
        },
    }
    a = Alert.from_json(payload)
    triage = build_triage(a, hits=[])
    assert "alert" in triage
    assert "hypotheses" in triage
    assert "first_steps" in triage
    assert "runbooks" in triage
    assert triage["alert"]["name"] == "HighErrorRate"
