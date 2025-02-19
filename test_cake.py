from EasyEdit.easyeditor import MEMITHyperParams,LoRAHyperParams,WISEHyperParams, ROMEHyperParams
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
from edit_utils import edit, edit_ifmet, edit_mello, edit_rome, cake, edit_wise, get_sent_embeddings
from EasyEdit.easyeditor.util.alg_dict import *
import torch
import os

def calculate_averages(data):
    total_cases = len(data)
    accuracy_true_count = [0, 0] 
    hop_wise_case_averages = [] 
    
    for case in data:
        hop_wise = case['post']['hop_wise']
        accuracy = case['post']['accuracy']
        case_hop_true_count = 0
        total_hops = 0
        for hop_pair in hop_wise:
            case_hop_true_count += sum(1 for x in hop_pair if x)
            total_hops += len(hop_pair)
        
        case_hop_average = case_hop_true_count / total_hops
        hop_wise_case_averages.append(case_hop_average)
        
        accuracy_true_count[0] += 1 if accuracy[0] else 0
        accuracy_true_count[1] += 1 if accuracy[1] else 0
    
    overall_hop_wise_avg = sum(hop_wise_case_averages) / total_cases
    accuracy_avg = [count/total_cases for count in accuracy_true_count]
    
    return {
        "total_cases": total_cases,
        "overall_hop_wise_average": f"{overall_hop_wise_avg:.3f}",
        "accuracy_averages": {
            "cloze": f"{accuracy_avg[0]:.3f}",
            "qa": f"{accuracy_avg[1]:.3f}"
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--datatype', default=None,type=str)
    parser.add_argument('--model_type', default=None,type=str)
    args = parser.parse_args()

    if args.editing_method == 'MEMIT' or args.editing_method == 'IFMET':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    else:
        editing_hparams = LoRAHyperParams
    if args.editing_method == 'CAKE' or args.editing_method == 'Mello':
        hparams=editing_hparams.from_hparams(f'./EasyEdit/hparams/LoRA/{args.model_type}.yaml')
    elif args.editing_method == 'IFMET':
        hparams=editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}-shallow.yaml')
        hparams_s=hparams
        hparams_d=editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}-deeper.yaml')
    else:
        hparams=editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}.yaml')
    
    if args.editing_method != 'Mello':
        alg_name = hparams.alg_name
        apply_algo = ALG_DICT[alg_name]

    MODEL_PATH = hparams.model_name
    if args.editing_method == 'CAKE':
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto",torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto",torch_dtype=torch.float32)
    # 添加FFN层的LoRA配置
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    import json
    data = json.load(open(f'./datasets/{args.datatype}-cake.json','r'))
    if args.editing_method == 'WISE':
        loc_data = json.load(open('./datasets/ZsRE/zsre_mend_train.json','r'))
        loc_data = loc_data[:7000]
        loc_index = 0
    elif args.editing_method == 'Mello':
        contriever = AutoModel.from_pretrained("facebook/contriever-msmarco", device_map=f"cuda:{hparams.device}")
        contriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        with open('./prompts/MeLLo-prompt.txt', 'r') as f:
            task_prompt = f.read()
        stop = ["Retrieved fact:"]
        new_facts = set()
        for d in data:
            for r in d["requested_rewrite"]:
                new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
        new_facts = list(new_facts)
        embs = get_sent_embeddings(new_facts, contriever, contriever_tokenizer, hparams.device)

    all_metrics = []
    for item in tqdm(data[:2]):
        if args.editing_method == 'CAKE':
            model, metrics = cake(model, tokenizer, item, hparams, test_generation=False)
        elif args.editing_method == 'WISE':
            metrics, loc_index = edit_wise(model, tokenizer, item, hparams, loc_data, loc_index, apply_algo, test_generation=False)
        elif args.editing_method == 'Mello':
            metrics = edit_mello(model, task_prompt, stop, tokenizer, item, hparams, contriever, contriever_tokenizer, embs, test_generation=False)
        elif args.editing_method == 'IFMET':
            model, metrics = edit_ifmet(model, tokenizer, item, hparams_s, hparams_d, apply_algo, test_generation=False)
        elif args.editing_method == 'ROME':
            model, metrics = edit_rome(model, tokenizer, item, hparams, apply_algo, test_generation=False)
        else:
            model, metrics = edit(model, tokenizer, item, hparams, alg_name, apply_algo, test_generation=False)
        print(metrics)
        all_metrics.append(metrics)
    res = calculate_averages(all_metrics)
    os.makedirs(args.metrics_save_dir,exist_ok=True)
    json.dump(all_metrics, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_metrics.json','w'),indent=4)
    json.dump(res, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_res.json','w'),indent=4)
