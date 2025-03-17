from datasets import load_dataset
import json
import random
random.seed(42)

def get_hellaswag():
    path = 'data/hellaswag'
    json = []
    dataset = load_dataset("Rowan/hellaswag", split='train', streaming=True)
    for i, data in enumerate(iter(dataset)):
        ctx = data['ctx']
        endings = data['endings']
        label = int(data['label'])
        inpt = f"""You are provided with an incomplete passage below as well as 4 endings in quotes and separated by commas, with only one of them being the correct ending. Treat the endings as being labelled 0, 1, 2, 3 in order. Please respond with the number corresponding to the correct ending for the passage.

### Passage: {ctx}

### Endings: {endings}"""
        json.append({'input':inpt, 'target':str(label), 'answer_choices':['0','1','2','3'], 'label':label})
    return path, json


def get_glue_qqp():
    path = 'data/glue_qqp'
    json = []
    dataset = load_dataset("nyu-mll/glue", "qqp", split='train', streaming=True)
    for i, data in enumerate(iter(dataset)):
        sentence1 = data['question1']
        sentence2 = data['question2']
        label = int(data['label'])
        inpt = f"""You are given two questions below, Question 1 and Question 2. If the two questions are semantically equivalent, please return 1. Otherwise, please return 0.\n\n### Question 1: {sentence1}\n\n### Question 2: {sentence2}"""
        json.append({'input':inpt, 'target':str(label), 'answer_choices':["0", "1"], 'label':label})
    return path, json

def get_glue_mrpc():
    path = 'data/glue_mrpc'
    json = []
    dataset = load_dataset("nyu-mll/glue", "mrpc", split='train', streaming=True)
    for i, data in enumerate(iter(dataset)):
        sentence1 = data['sentence1']
        sentence2 = data['sentence2']
        label = int(data['label'])
        inpt = f"""You are given two sentences below, Sentence 1 and Sentence 2. If the two sentences are semantically equivalent, please return 1. Otherwise, please return 0.\n\n### Sentence 1: {sentence1}\n\n### Sentence 2: {sentence2}"""
        json.append({'input':inpt, 'target':str(label), 'answer_choices':["0", "1"], 'label':label})
    return path, json

def get_glue_cola():
    path = 'data/glue_cola'
    json = []
    dataset = load_dataset("nyu-mll/glue", "cola", split='train', streaming=True)
    for i, data in enumerate(iter(dataset)):
        sentence = data['sentence']
        label = int(data['label'])
        inpt = f"""Determine if the sentence below is syntactically and semantically correct. If it is syntactically and semantically correct, respond "1". Otherwise, respond "0".\n\nSentence:{sentence}"""
        json.append({'input':inpt, 'target':str(label), 'answer_choices':["0", "1"], 'label':label})
    return path, json

def get_glue_sst2():
    path = 'data/glue_sst2.json'
    json = []
    dataset = load_dataset("nyu-mll/glue", "sst2", split='train', streaming=True)
    for i, data in enumerate(iter(dataset)):
        sentence = data['sentence']
        label = int(data['label'])
        inpt = f"""Given the following sentence:\n\n{sentence}

Respond with 0 if the sentiment of the sentence is negative and 1 if the sentiment of the sentence is positive."""
        json.append({'input':inpt, 'target':str(label), 'answer_choices':["0", "1"], 'label':label})
    return path, json

def get_glue_mnli():
    path = 'data/glue_mnli'
    json = []
    dataset = load_dataset("nyu-mll/glue", "mnli", split='train', streaming=True)
    for i, data in enumerate(iter(dataset)):
        premise = data['premise']
        hypo = data['hypothesis']
        label = int(data['label'])
        inpt = f"""You are given a premise and a hypothesis below. If the premise entails the hypothesis, return 0. If the premise contradicts the hypothesis, return 2. Otherwise, if the premise does neither, return 1.

### Premise: {premise}\n\n### Hypothesis: {hypo}"""
        json.append({'input':inpt, 'target':str(label), 'answer_choices':["0", "1", "2"], 'label':label})
    return path, json

def get_glue_qnli():
    path = 'data/glue_qnli'
    json = []
    dataset = load_dataset("nyu-mll/glue", "qnli", split='train', streaming=True)
    for i, data in enumerate(iter(dataset)):
        question = data['question']
        sentence = data['sentence']
        label = int(data['label'])
        inpt = f"""You are provided a question and a corresponding response below. If the response properly answers the question, please return 0. Otherwise, please return 1.

### Question: {question}\n\n### Response: {sentence}"""
        json.append({'input':inpt, 'target':str(label), 'answer_choices': ["0", "1"], 'label': label})
    return path, json

def get_agnews():
    path = 'data/agnews'
    json = []
    dataset = load_dataset("fancyzhx/ag_news", split='train', streaming=True)
    for i, data in enumerate(iter(dataset)):
        choices = ['World', 'Sports', 'Business', 'Sci/Tech']
        content = data['text']
        label = int(data['label'])
        inpt = f"""Below is a news article. Please classify it under one of the following classes (World, Business, Sports, Sci/Tech)

### Article: {content}"""
        json.append({"input":inpt, "target":choices[label], "answer_choices":choices, "label":label})
    return path, json


def get_dbpedia():
    path = 'data/dbpedia'
    json = []
    dataset = load_dataset('fancyzhx/dbpedia_14', split='train',streaming=True)
    for i, data in enumerate(iter(dataset)):
        title = data['title']
        content = data['content']
        label = int(data['label'])
        choices = ['Company', 'Educational Institution', 'Artist', 'Athlete', 'Office Holder', 'Mean of Transportation', 'Building', 'Natural Place', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'Written Work']
        inpt = f"""You are given the title and the body of an article below. Please determine the type of the article.
### Title: {title}
 
### Body: {content}

### Choices: {', '.join(choices)}"""
        target = choices[label]
        json.append({'input':inpt, 'target':target, 'answer_choices':choices, 'label': label})
    return path, json


for dataset in [get_hellaswag, get_glue_qqp, get_glue_mrpc, get_glue_cola, get_glue_sst2, get_glue_mnli, get_glue_qnli, get_agnews, get_dbpedia]:
    # get data
    name, data = dataset()
    
    # create splits
    random.shuffle(data)
    split = 2 * len(data)//3
    print(f"Writing {name}...")
    for i,s in enumerate(['train', 'test']):
        with open(name+f'_{s}.json', 'w') as fd:
            dat = data[:split] if i == 0 else data[split:]
            for d in dat:
                json.dump(d, fd)
                fd.write('\n')
