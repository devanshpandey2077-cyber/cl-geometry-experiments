"""Sequential, SINGLE-adapter LoRA with held-out evaluations and hidden-state probes.
Example: python train_and_probe.py --data data --output runs/run1 --max-steps 400
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

TASKS=['sst2','qqp','qnli','mnli','ag_news']
def load_jsonl(path):
    with open(path,encoding='utf8') as f:return [json.loads(line) for line in f]

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data');p.add_argument('--output',default='runs/pilot')
    p.add_argument('--model',default='Qwen/Qwen3-1.7B-Base')
    p.add_argument('--order',default=','.join(TASKS));p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rank',type=int,default=16);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--accum',type=int,default=8)
    p.add_argument('--max-steps',type=int,default=400,help='optimizer steps per task')
    p.add_argument('--max-length',type=int,default=256)
    p.add_argument('--probes',type=int,default=500,help='fixed subset of probe file for pilot; can use 2000')
    p.add_argument('--eval-limit',type=int,default=1000)
    p.add_argument('--layers',default='6,14,22',help='hidden_states tuple indices, 0=embedding')
    p.add_argument('--frozen',action='store_true',help='Only probe the original frozen model; no fine-tuning')
    args=p.parse_args()
    order=args.order.split(',')
    assert len(set(order))==len(order) and set(order)==set(TASKS), 'Order must include five tasks once'
    set_seed(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    tok.padding_side='right'
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if not args.frozen:
        lora=LoraConfig(r=args.rank,lora_alpha=args.rank*2,lora_dropout=0.0,target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',task_type='CAUSAL_LM')
        model=get_peft_model(model,lora)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    layers=[int(x) for x in args.layers.split(',')]
    print('Device:',device,'layers:',layers)
    metrics=[]

    def eval_task(task, limit):
        rows=load_jsonl(Path(args.data)/task/'test.jsonl')[:limit]
        model.eval();correct=0;loss_sum=0
        # Classification by conditional log-likelihood of each allowed answer token.
        label_options=sorted({r['answer'] for r in rows})
        with torch.no_grad():
            for row in rows:
                prompt=row['input']
                prompt_ids=tok(prompt,add_special_tokens=False,return_tensors='pt')['input_ids'].to(device)
                # Decoder predicts token following last prompt token.
                logits=model(input_ids=prompt_ids).logits[0,-1].float().log_softmax(dim=-1)
                scores={label:logits[tok.encode(label,add_special_tokens=False)[0]].item() for label in label_options}
                predicted=max(scores,key=scores.get)
                correct+=predicted==row['answer']
        return correct/len(rows)

    def probe(stage):
        model.eval(); geometry=out/'features'/f'stage_{stage}';geometry.mkdir(parents=True,exist_ok=True)
        with torch.no_grad():
            for task in TASKS:
                rows=load_jsonl(Path(args.data)/task/'probe.jsonl')[:args.probes]
                layer_records={i:[] for i in layers}
                for row in rows:
                    # Full prompt (fixed) pooled over all non-padding prompt tokens.
                    encoding=tok(row['input'],return_tensors='pt',truncation=True,max_length=args.max_length).to(device)
                    res=model(**encoding,output_hidden_states=True,use_cache=False)
                    for i in layers:
                        h=res.hidden_states[i][0].float().mean(dim=0)
                        layer_records[i].append(h.cpu().numpy())
                for i,vectors in layer_records.items():
                    np.save(geometry/f'{task}_layer{i}.npy',np.stack(vectors).astype('float32'))
                print('Probed',stage,task,len(rows))

    def metrics_at(stage, trained):
        vals={task:eval_task(task,args.eval_limit) for task in trained}
        metrics.append({'stage':stage,'trained':list(trained),'accuracy':vals})
        (out/'accuracy.json').write_text(json.dumps(metrics,indent=2))
        print('Evaluation',stage,vals)

    probe(0)
    if args.frozen:
        print('Frozen control complete: geometry is stage 0; repeat identical probes to test determinism.');return
    trained=[]
    for stage,task in enumerate(order,1):
        print('Training task',task)
        rows=load_jsonl(Path(args.data)/task/'train.jsonl')
        random.Random(args.seed+stage).shuffle(rows)
        model.train()
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr)
        gradsteps=0;microsteps=0;cursor=0;opt.zero_grad()
        while gradsteps<args.max_steps:
            batch=rows[cursor:cursor+args.batch]
            cursor+=args.batch
            if not batch:cursor=0;continue
            encoded=[]
            for row in batch:
                pids=tok.encode(row['input'],add_special_tokens=False)
                answer=tok.encode(row['answer'],add_special_tokens=False)
                if len(answer)!=1: raise ValueError(f'Label {row["answer"]} must be one token with this tokenizer')
                # Reserve one position for label; supervision on answer only
                pids=pids[-(args.max_length-1):]
                ids=pids+answer
                encoded.append((ids,[-100]*len(pids)+answer))
            maxlen=max(len(x[0]) for x in encoded)
            ids=torch.tensor([x[0]+[tok.pad_token_id]*(maxlen-len(x[0])) for x in encoded],device=device)
            labels=torch.tensor([x[1]+[-100]*(maxlen-len(x[1])) for x in encoded],device=device)
            attn=torch.tensor([[1]*len(x[0])+[0]*(maxlen-len(x[0])) for x in encoded],device=device)
            loss=model(input_ids=ids,attention_mask=attn,labels=labels,use_cache=False).loss/args.accum
            loss.backward();microsteps+=1
            if microsteps%args.accum==0:
                opt.step();opt.zero_grad();gradsteps+=1
                if gradsteps%100==0: print('step',gradsteps,'loss',float(loss.item()*args.accum))
        trained.append(task)
        ckpt=out/'checkpoints'/f'stage_{stage}_{task}';ckpt.mkdir(parents=True,exist_ok=True)
        model.save_pretrained(ckpt)
        metrics_at(stage,trained)
        probe(stage)
    print('Finished:',out)

if __name__=='__main__':main()
