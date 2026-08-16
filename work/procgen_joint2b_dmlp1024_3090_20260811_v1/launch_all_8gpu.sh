#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/procgen_joint2b_dmlp1024_20260811_v1

launch_queue() {
    local gpu=$1
    shift
    local log="$ROOT/queue_gpu${gpu}.log"
    setsid nohup "$ROOT/run_gpu_queue.sh" "$gpu" "$@" > "$log" 2>&1 < /dev/null &
    echo "$!" > "$ROOT/queue_gpu${gpu}.pid"
}

# Seeds 0 and 1 start concurrently across the four matched environments.
# Seed 2 continues on GPUs 0-3 after the corresponding seed-0 run finishes.
launch_queue 0 bigfish-easy-0-10 0 bigfish-easy-0-10 2
launch_queue 1 bossfight-easy-0-10 0 bossfight-easy-0-10 2
launch_queue 2 caveflyer-easy-0-10 0 caveflyer-easy-0-10 2
launch_queue 3 coinrun-easy-0-10 0 coinrun-easy-0-10 2
launch_queue 4 bigfish-easy-0-10 1
launch_queue 5 bossfight-easy-0-10 1
launch_queue 6 caveflyer-easy-0-10 1
launch_queue 7 coinrun-easy-0-10 1

sleep 2
for gpu in 0 1 2 3 4 5 6 7; do
    pid=$(tr -d '\n' < "$ROOT/queue_gpu${gpu}.pid")
    kill -0 "$pid"
    echo "GPU=$gpu QUEUE_PID=$pid"
done
