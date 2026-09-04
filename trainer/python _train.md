python train\_grpo.py --use\_moe=0 --reasoning=1 --from\_resume=1 --loss\_type=grpo



python eval.py --load\_from model --weight grpo --save\_dir ../out --hidden\_size 512 --num\_hidden\_layers 8 --use\_moe 0 --max\_new\_tokens 1536 --temperature 0.8



python train\_agent.py --hidden\_size 512 --num\_hidden\_layers 8 --use\_moe 0 --from\_weight reason --save\_weight agent --save\_dir ../out --data\_path ../dataset/agent\_rl.jsonl --loss\_type grpo --num\_workers 0



python train\_agent.py --from\_weight reason --loss\_type grpo --num\_workers 0

