"""Scratch file for debugging"""
import torch
import transformers
from datasets import load_dataset_builder
from datasets import load_dataset
from lm_eval.models.dist_enc_utils import ParameterlessAttentionDecoder

model =transformers.AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
decoder = ParameterlessAttentionDecoder(model)
model.eval()
decoder.eval()
dataset = {'alpaca_train': load_dataset("tatsu-lab/alpaca", split='train')}
dataset: list = dataset['alpaca_train']

with torch.no_grad():
    for i in range(10000):
