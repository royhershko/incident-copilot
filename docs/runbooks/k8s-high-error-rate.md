# K8s High Error Rate / 5xx Spike

## Symptoms
- Error rate increases, 5xx spikes, latency jumps, timeouts.
- Often correlated with deploys or config changes.

## Immediate Checks
- Rollout history: `kubectl -n <ns> rollout history deploy/<name>`
- Pods health: `kubectl -n <ns> get pods -o wide`
- Recent events: `kubectl -n <ns> get events --sort-by=.metadata.creationTimestamp | tail -n 50`
- Logs sample: `kubectl -n <ns> logs deploy/<name> --since=10m | tail -n 200`

## Likely Causes
- Bad deploy / image / config
- Dependency failures (DB/Redis/3rd party)
- Resource saturation (CPU/Mem/threads/conn pool)

## Mitigation
- Rollback: `kubectl -n <ns> rollout undo deploy/<name>`
- Temporary scale (if safe): `kubectl -n <ns> scale deploy/<name> --replicas=<n>`
