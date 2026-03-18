

if [ -z "$1" ]; then
    echo "Usage: $0 <hostfile>"
    exit 1
fi
HOSTFILE=$1

source ../env.sh > /dev/null 2>&1

python ../scripts/wait_for_vllm_servers.py \
	--hostfile "$HOSTFILE" \
	--output ../scripts/swarm.hostfile

python ./lab3_semantic_uncertainty.py \
  --hostfile ../scripts/swarm.hostfile \
  --prompts ../prompts/nmnat2_peptide_inhibitors.txt \
  --num-samples 20 \
  --method kle
