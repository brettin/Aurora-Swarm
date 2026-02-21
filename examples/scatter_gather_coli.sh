python ../scripts/wait_for_vllm_servers.py --hostfile ../scripts/hostfile --output ../scripts/swarm.hostfile

python ../examples/scatter_gather_coli.py /home/brettin/ModCon/brettin/Aurora-Inferencing/examples/TOM.COLI/batch_1/ --hostfile ../scripts/swarm.hostfile --num-files 16 --output test.openai.batch.txt --model openai/gpt-oss-120b --batch-size 2000
