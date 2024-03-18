import os
import time
import copy
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from huggingface_hub import login
import accelerate
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

experiment = {"NUM_CLASSES" : 6}

class SimplestLinearHead(nn.Module):
    def __init__(self, lm_output_size:int, num_classes:int):
        super(SimplestLinearHead, self).__init__()
        self.fc = nn.Linear(lm_output_size, num_classes)

    def forward(self, lm_hidden_states):
        pooled_output = torch.mean(lm_hidden_states, dim=1)
        logits = self.fc(pooled_output)
        return logits

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
access_token = "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"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '</s>'})

lm = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token, quantization_config=bnb_config)

classifier = SimplestLinearHead(lm.config.hidden_size, experiment["NUM_CLASSES"]).to(device)

previous_checkpoint_file = "ENTER .pth FILE HERE"
previous_checkpoint = torch.load(previous_checkpoint_file)
classifier.load_state_dict(previous_checkpoint['classifier_state_dict'])



classifier.eval()
lm.eval()

input_string = "ENTER THE STRING HERE"
tokenized_input = tokenizer(input_string)
lm_outputs = lm(tokenized_input["input_ids"])
classifier_outputs = classifier(lm_outputs[0].float())

# These classifier outputs are the logits
# A call to torch.softmax(classifier_outputs) should do it! 

# to get probabilities for each label:
label_probs = torch.softmax(classifier_outputs)

#to get the single most probable label:
most_probable = classifier_outputs.argmax(dim=1)
