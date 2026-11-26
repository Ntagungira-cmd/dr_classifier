#!/usr/bin/env bash
# Flood /predict with the same Locust load against 1, 2 and 3 API containers.
# Results: locust_results/api<N>_stats.csv (+ _stats_history.csv, _failures.csv)
#
# Usage: scripts/run_locust_scaling.sh [users] [duration]   (default 30 users, 60s)
set -euo pipefail
cd "$(dirname "$0")/.."

USERS="${1:-30}"
DURATION="${2:-60s}"
mkdir -p locust_results

wait_healthy() {
  local want=$1
  for _ in $(seq 1 100); do
    local n
    n=$(docker compose ps api --format '{{.Health}}' | grep -c '^healthy$' || true)
    [ "$n" -ge "$want" ] && return 0
    sleep 3
  done
  echo "timed out waiting for $want healthy api containers" >&2
  exit 1
}

for N in 1 2 3; do
  echo "=== $N API container(s), $USERS users, $DURATION ==="
  docker compose up -d --scale api="$N" --no-recreate api ui nginx >/dev/null
  wait_healthy "$N"
  # nginx resolves "api" once at startup; restart so it balances across every replica
  docker compose restart nginx >/dev/null
  sleep 5
  # --no-deps: otherwise "run" re-applies the default scale and drops the extra api replicas
  docker compose --profile load run --rm --no-deps locust \
    locust -f locust/locustfile.py --host http://nginx:80 --headless \
    -u "$USERS" -r 10 -t "$DURATION" --csv "locust_results/api${N}" --only-summary 2>&1 | tail -4
  echo "requests handled per container:"
  for c in $(docker compose ps -q api); do
    echo "  $(docker inspect -f '{{.Name}}' "$c"): $(docker logs --since "${DURATION}" "$c" 2>&1 | grep -c 'POST /predict')"
  done
done

docker compose up -d --scale api=1 api ui nginx >/dev/null
docker compose restart nginx >/dev/null
echo
echo "containers,requests,failures,rps,median_ms,avg_ms,p95_ms,p99_ms,max_ms"
for N in 1 2 3; do
  awk -F, -v n="$N" '$2=="Aggregated"{printf "%s,%s,%s,%.1f,%s,%.0f,%s,%s,%.0f\n", n,$3,$4,$10,$5,$6,$17,$19,$8}' \
    "locust_results/api${N}_stats.csv"
done
