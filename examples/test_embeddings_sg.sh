#!/usr/bin/env bash
# Test embeddings API using scatter-gather (default model: sentence-transformers/all-MiniLM-L6-v2).
# Usage: ./test_embeddings_sg.sh <hostfile>
# Run from the examples/ directory. Waits for vLLM servers, then scatters
# a few test strings across agents and gathers embedding results.

if [ -z "$1" ]; then
    echo "Usage: $0 <hostfile>"
    exit 1
fi
HOSTFILE=$1

python ../scripts/wait_for_vllm_servers.py --hostfile "$HOSTFILE" --output ../scripts/swarm.hostfile

python ./test_embeddings_sg.py --hostfile ../scripts/swarm.hostfile
