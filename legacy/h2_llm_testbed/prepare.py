"""Prepare disjoint, balanced NLP CL splits. Does not fabricate examples."""
import argparse, json, hashlib
from collections import Counter
from pathlib import Path
from datasets import load_dataset
import numpy as np

TASKS = {
    'sst2': ('nyu-mll/glue', 'sst2', ['sentence'], {0:'A',1:'B'}),
    'qqp': ('nyu-mll/glue', 'qqp', ['question1','question2'], {0:'A',1:'B'}),
    'qnli': ('nyu-mll/glue', 'qnli', ['question','sentence'], {0:'A',1:'B'}),
    'mnli': ('nyu-mll/glue', 'mnli', ['premise','hypothesis'], {0:'A',1:'B',2:'C'}),
    'ag_news': ('fancyzhx/ag_news', None, ['text'], {0:'A',1:'B',2:'C',3:'D'})
}
GUIDES = {
    'sst2':'Classify sentiment. A = negative; B = positive.',
    'qqp':'Are these questions duplicates? A = no; B = yes.',
    'qnli':'Does the sentence contain the answer to the question? A = entailment; B = not entailment.',
    'mnli':'Classify the relation. A = entailment; B = neutral; C = contradiction.',
    'ag_news':'Classify news topic. A = world; B = sports; C = business; D = science/technology.'
}

def canonical_fields(row, keys):
    out=[]
    for key in keys:
        value=row.get(key)
        if not isinstance(value,str) or not value.strip(): return None
        out.append(' '.join(value.split()))
    return out

def make_input(task, fields):
    if task=='sst2': content=f'Text: {fields[0]}'
    elif task=='qqp': content=f'Question 1: {fields[0]}\nQuestion 2: {fields[1]}'
    elif task=='qnli': content=f'Question: {fields[0]}\nSentence: {fields[1]}'
    elif task=='mnli': content=f'Premise: {fields[0]}\nHypothesis: {fields[1]}'
    else: content=f'Text: {fields[0]}'
    return f'{GUIDES[task]}\n{content}\nAnswer:'

def write_jsonl(path,rows):
    with open(path,'w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='data')
    p.add_argument('--train',type=int,default=50000)
    p.add_argument('--validation',type=int,default=2000)
    p.add_argument('--probe',type=int,default=2000)
    p.add_argument('--test',type=int,default=5000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--allow-smaller',action='store_true',help='explicitly allow less than requested, never replacement sampling')
    args=p.parse_args()
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    counts={k:getattr(args,k) for k in ['train','validation','probe','test']}
    manifest={'seed':args.seed,'requested':counts,'tasks':{},'warning':'These are custom splits of public labeled training data, NOT official benchmark test scores. No cross-task near-duplicate guarantee.'}
    # Fingerprints shared across tasks prevent EXACT identical content from leaking across task splits.
    # Crucial: task-dependent prompts are NOT part of the fingerprint.
    global_seen={}
    all_rows={}
    for task,(repo,config,keys,mapping) in TASKS.items():
        ds=load_dataset(repo,config,split='train') if config else load_dataset(repo,split='train')
        candidates=[]; local_seen=set()
        for row in ds:
            label=row.get('label')
            if label not in mapping:continue
            fields=canonical_fields(row,keys)
            if fields is None:continue
            fingerprint=hashlib.sha256(json.dumps(fields,ensure_ascii=False,sort_keys=True).lower().encode()).hexdigest()
            if fingerprint in local_seen: continue
            local_seen.add(fingerprint)
            candidates.append({'task':task,'input':make_input(task,fields),'answer':mapping[label], 'label':int(label),'source_id':str(row.get('idx',fingerprint[:16])),'fingerprint':fingerprint})
        all_rows[task]=candidates
        print(f'{task}: {len(candidates)} unique labeled candidate rows')
    # Assign exact cross-task collisions to first encountered task; prevents same-text leakage across tasks.
    for task in TASKS:
        filtered=[]
        for row in all_rows[task]:
            key=row['fingerprint']
            if key not in global_seen:
                global_seen[key]=task;filtered.append(row)
        all_rows[task]=filtered
    for task,rows in all_rows.items():
        by_label={k:[] for k in TASKS[task][3]}
        for row in rows: by_label[row['label']].append(row)
        for label,items in by_label.items(): rng.shuffle(items)
        cursor={label:0 for label in by_label}; output_dir=root/task;output_dir.mkdir(exist_ok=True)
        actual={}; n_classes=len(by_label)
        for split,n in counts.items():
            per=n//n_classes
            remainder=n%n_classes
            if any(len(items)-cursor[label]<per for label,items in by_label.items()):
                available=min(len(items)-cursor[label] for label,items in by_label.items())
                if not args.allow_smaller:
                    raise ValueError(f'{task}/{split}: need {per}/class; at most {available}/class remain. Reduce sizes or use --allow-smaller.')
                per=max(0,available)
            selected=[]
            for position,(label,items) in enumerate(by_label.items()):
                take=per+(1 if position<remainder and per==n//n_classes else 0)
                # When allow-smaller applies, use only the feasible balanced count.
                if per<n//n_classes: take=per
                take=min(take,len(items)-cursor[label])
                selected+=items[cursor[label]:cursor[label]+take]
                cursor[label]+=take
            rng.shuffle(selected)
            if not selected:raise ValueError(f'No examples for {task}/{split}')
            write_jsonl(output_dir/f'{split}.jsonl',selected)
            actual[split]={'size':len(selected),'label_counts':dict(Counter(str(x['label']) for x in selected))}
        manifest['tasks'][task]={'source':TASKS[task][0], 'config':TASKS[task][1], 'actual':actual,'label_tokens':TASKS[task][3]}
        print(task,actual)
    with open(root/'manifest.json','w') as f:json.dump(manifest,f,indent=2)
    print('Done:',root)

if __name__=='__main__': main()
