module load frameworks
conda activate /lus/flare/projects/ModCon/brettin/conda_envs/swarm


python ./scripts/wait_for_vllm_servers.py --hostfile /home/brettin/ModCon/brettin/develop/Aurora-Inferencing/vllm-gpt-oss120b/hostfile --output /home/brettin/ModCon/brettin/develop/Aurora-Inferencing/vllm-gpt-oss120b/running.hostfile --health-timeout 600


python ./examples/scatter_gather_coli.py /home/brettin/ModCon/brettin/Aurora-Inferencing/examples/TOM.COLI/batch_1/ --hostfile /home/brettin/ModCon/brettin/develop/Aurora-Inferencing/vllm-gpt-oss120b/hostfile --num-files 16 --output test.openai.batch.txt --model openai/gpt-oss-120b --batch-size 2000
