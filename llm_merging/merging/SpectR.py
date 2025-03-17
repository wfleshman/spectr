import torch 
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from llm_merging.merging.Merges import Merges
from llm_merging.model.decoder_functions import tokenize_batch
import os
import gc
from pathlib import Path
from string import ascii_letters
ascii_letters = ascii_letters[-26:]

def svd(A, B, niter=50):
    torch.random.manual_seed(42)
    rank = A.shape[0]
    with torch.no_grad():
        U,S,V = torch.svd_lowrank((B@A).type(torch.float32), q=rank, niter=niter)
        B = U[:, :rank]@torch.diag(S[:rank])
        A = V.T[:rank,:]
        return A.type(torch.bfloat16), B.type(torch.bfloat16)
    
class RouteWrap(torch.nn.Module):

    def __init__(self, name, module, adptrs, model, topk=4):
        super().__init__()
        self.name = None
        self.model = model
        self.module = module
        self.topk = topk
        p_name = [p for p in list(adptrs.keys()) if name in p]
        if len(p_name) == 0:
            return
  
        adapters = adptrs[p_name[0]]
        name = name + '.weight'
        self.name = name
        self.module = module
        with torch.no_grad():
            if isinstance(module, torch.nn.Linear):
                self.A = torch.nn.Parameter(torch.concat([a[0][None] for a in adapters], dim=0))
                self.B = torch.nn.Parameter(torch.concat([b[1][None] for b in adapters], dim=0))
    
    def forward(self, x):
        if self.name is None:
            return self.module(x)
        if isinstance(self.module, torch.nn.Linear):
            vecs = torch.einsum('arh,bsh->absr', self.A, x)
            arrows = torch.norm(vecs, dim=-1) # abs
            _, idx = torch.topk(arrows, self.topk, dim=0)
                        
            routes = torch.zeros_like(arrows)
            routes.scatter_(0, idx, 1/self.topk)
            vecs = (routes[...,None] * vecs).sum(0) # bsr
            B = torch.einsum('abs,ajr->bsjr', routes, self.B)
            delta = torch.einsum('bsjr,bsr->bsj', B, vecs)
            return self.module(x) + delta

        return self.module(x)
    
def format_example(inpt, choices, tokenizer):
    prompt = inpt + '\n\n### Options:\n'+'\n'.join([f'{ascii_letters[i]} : {choices[i]}' for i in range(len(choices))])
    answer = "### Correct Answer:\n"
    messages = [
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + answer
   

class SpectR(Merges):
    def __init__(self, name):
        super().__init__(name)
        set_seed(42)

        # Model info
        self.base_model_name = os.environ["EVAL_MODEL"]
        
        model_dir = os.path.join(os.environ["MODEL_DIR"], self.base_model_name)
        self.list_models = [str(x) for x in Path(model_dir).glob('*/checkpoint*')]

        self.max_seq_len = 512

        # Architecture must match base model. 
        self.architecture = "decoder"
        
        # LoRA scaling
        self.scaling = 2
        
        # Loaded models and configs 
        self.loaded_models = {}
        self.loaded_configs = {}

        # Merged model parameters
        self.merged_model = {}

    def predict_multiple_choice(self, batch):
        choices = batch['answer_choices'][0]
        inpt = batch["input"][0]
        batch['input'] = [
            format_example(inpt, choices, self.input_tokenizer)
        ]
        tokenized_batch = tokenize_batch(self.input_tokenizer, self.target_tokenizer, batch, self.base_model.device)
        
        output = self.base_model(
            input_ids=tokenized_batch["input_ids"],
            attention_mask=tokenized_batch["input_mask"],
        )
                
        choice_ids = self.target_tokenizer.convert_tokens_to_ids(list(ascii_letters)[:len(choices)])
        scores = output.logits[0,-1][choice_ids]
        _,choice = torch.max(scores, dim=0)
        return choice.unsqueeze(0), scores.unsqueeze(0).float()
    
    def _load_base_model(self):
        # Load
        self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, device_map='auto',torch_dtype=torch.bfloat16, token=os.environ["HF_AUTH_TOKEN"])
        self.base_model.resize_token_embeddings(len(self.input_tokenizer), pad_to_multiple_of=8)


    def _load_tokenizer(self):
        self.input_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, token=os.environ["HF_AUTH_TOKEN"])
        self.input_tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.input_tokenizer.padding_side = "right"
        self.input_tokenizer.model_max_length = 512
        self.target_tokenizer = self.input_tokenizer
        
    def merge(
        self,
    ):
        # 1. Load base model and tokenizer
        self._load_tokenizer()
        self._load_base_model()

        # 2. Load additional models and align
        super()._load_huggingface_models_and_configs()    
        all_models = list(self.loaded_models.values())
        all_parameter_names = all_models[0].keys()

        adapters = {name:[] for name in all_parameter_names if 'A' in name}
        for parameter_name in all_parameter_names:
            if 'A' in parameter_name:
                other_name = list(parameter_name)
                other_name[other_name.index('A')] = 'B'
                other_name = ''.join(other_name)
                for model in all_models:
                    A = model[parameter_name]
                    B = model[other_name]
                    A, B = svd(A, B)
                    adapters[parameter_name].append((self.scaling * A, B))            

        # 3. Enable per token per layer routing between adapters
        modules = {name:module for name,module in self.base_model.named_modules()}
        for name, module in modules.items():
            if hasattr(module, 'weight'):
                parent = self.base_model
                for attr in name.split('.')[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, name.split('.')[-1], RouteWrap(name, module, adapters, self))
        del adapters
        gc.collect()
        torch.cuda.empty_cache()

        # Requires to make results deterministic. If not set, we will just run once and use the results from the first pass. 
        self.base_model.eval()

        return self.base_model
    
    def save_model(self, output_dir):
        return