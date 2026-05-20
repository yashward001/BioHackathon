#!/usr/bin/env bash
# run.sh — start the PolyClear ML API and open the UI in the browser.
set -e
cd "$(dirname "$0")"

# Kill anything already bound to port 5050
lsof -ti :5050 | xargs kill -9 2>/dev/null || true

echo "Starting PolyClear ML API on http://localhost:5050 ..."
.venv/bin/python3 api.py &
API_PID=$!

# Wait up to 5 s for the health endpoint to respond
for i in $(seq 1 10); do
  if curl -s http://localhost:5050/health >/dev/null 2>&1; then
    echo "API ready."
    break
  fi
  sleep 0.5
done

echo "Opening index.html ..."
open index.html

echo ""
echo "Press Ctrl+C to stop the API (PID $API_PID)."
wait $API_PID
