#!/usr/bin/env bash
# Launch the broadcast + aggregators example.
# Usage: ./broadcast_aggregators.sh <hostfile>
# Run from the examples/ directory. Waits for vLLM servers, then broadcasts
# to all hosts except one and runs majority_vote and concat aggregators.

HOSTFILE=$1

python ../scripts/wait_for_vllm_servers.py --hostfile "$HOSTFILE" --output ../scripts/swarm.hostfile

python ./broadcast_aggregators.py --hostfile ../scripts/swarm.hostfile --model openai/gpt-oss-120b --show-failures
