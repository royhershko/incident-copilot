# Incident Copilot (MVP)

A minimal SRE-style incident triage CLI.

The tool ingests an alert payload (Prometheus-style or generic JSON),
matches relevant runbooks, and produces a structured, human-readable triage.

## What it does
- Parses alert context (severity, service, namespace)
- Generates top incident hypotheses
- Suggests first-response investigation steps (read-only)
- Matches relevant Markdown runbooks with a relevance score
- Outputs both terminal-friendly view and JSON

## Quickstart

### Requirements
- Python 3.10+

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
