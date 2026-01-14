import argparse
import os
import json
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Data preprocessing")
    parser.add_argument("--train_data", type=str, required=False, default=None)
    parser.add_argument("--dealed_train_data", type=str, required=False, default=None)
    parser.add_argument("--prompt_path", type=str, required=False, default=None)
    parser.add_argument("--val_data", type=str, required=False, default=None)
    parser.add_argument("--dealed_val_data", type=str, required=False, default=None)
    return parser.parse_args()
def deal_data(prompt_path,data,dealed_path):
    with open(dealed_path,'w',encoding='utf-8') as f:
        pass
    with open(prompt_path,'r',encoding='utf-8') as f:
        template=f.read()
    dataset = data
    prompts=[]
    for i,question in enumerate(dataset['question']):
        #将question中的「question「替换为输入内容
        prompt=template.format(question=question)
        ground_truth=dataset['answer'][i]
        think,answer=ground_truth.split('####')
        dealed_ground_truth=think.strip()+' </think> <answer> '+answer.strip()+' </answer>'
        #处理ground_truth中的格式，使得其符合r1_zero_reward_fn的输入格式
        with open(dealed_path,'a',encoding='utf-8') as f:
            f.write(json.dumps({'prompt':prompt,'ground_truth':dealed_ground_truth,"correct_answer":answer.strip()},ensure_ascii=False)+'\n')
def main():
    args = parse_args()
    config_path = "/root/assignment5-alignment/cs336_alignment/config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
    data = load_dataset("json", data_files={"train": args.train_data, "test": args.val_data})
    train_dataset = data["train"]
    val_dataset = data["test"]
    deal_data(args.prompt_path, train_dataset, args.dealed_train_data)
    deal_data(args.prompt_path, val_dataset, args.dealed_val_data)
if __name__ == "__main__":
    main()
