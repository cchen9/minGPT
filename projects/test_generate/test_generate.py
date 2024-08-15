import os
os.environ['HF_HOME'] = '/home/ubuntu/USERS/clareche/cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
set_seed(3407)

use_mingpt = True
model_type = 'gpt2-xl'
device='cuda:1'

if use_mingpt:
    model = GPT.from_pretrained(model_type)
else:
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.config.pad_token_id = model.config.eos_token_id # suppress a warning

# ship the model to device and set to eval mode
model.to(device)
model.eval()
