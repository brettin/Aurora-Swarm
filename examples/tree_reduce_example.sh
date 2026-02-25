#!/usr/bin/env bash
# Launch the tree-reduce pattern example.
# Usage: ./tree_reduce_example.sh <hostfile>
# Run from the examples/ directory. Waits for vLLM servers, then runs
# hierarchical tree-reduce (leaf agents → supervisor summarization → single result).

if [ -z "$1" ]; then
    echo "Usage: $0 <hostfile>"
    exit 1
fi
HOSTFILE=$1

# Run from examples/; env.sh is at project root
source ../env.sh

python ../scripts/wait_for_vllm_servers.py --hostfile "$HOSTFILE" --output ../scripts/swarm.hostfile

python ./tree_reduce_example.py --hostfile ../scripts/swarm.hostfile --model openai/gpt-oss-120b --show-failures
