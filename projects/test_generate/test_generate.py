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

def generate(prompt='', num_samples=10, steps=20, do_sample=True):
    # tokenize the input prompt into integer input sequence
    if use_mingpt:
        tokenizer = BPETokenizer()
        if prompt == '':
            # to create unconditional samples...
            # manually create a tensor with only the specia <|endoftext|> token
            # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
            x = tokenizer.encoder.encoder['<|endoftext|>']
            print(x)
generate(prompt='', num_samples=10, steps=20)
