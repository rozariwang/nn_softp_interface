#import os
#import time
#import copy
#import pandas as pd
#import numpy as np
#import pickle
import torch
import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
#from torch.utils.data import DataLoader
#from datasets import Dataset, load_dataset
#from huggingface_hub import login
#import accelerate
from transformers import AutoModel,  AutoTokenizer
#from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
#import matplotlib.pyplot as plt

import streamlit as st


previous_checkpoint_file = "streamlit_BERT_binary_SimpleLinearHead_1710938476.406034/best_checkpoint_streamlit_BERT_binary_SimpleLinearHead_1710938476.406034.pth"

@st.cache_resource  # 👈 Add the caching decorator
def load_model() -> object:
    """

    """
    return AutoModel.from_pretrained("bert-base-uncased")

@st.cache_resource  # 👈 Add the caching decorator
def load_tokenizer() -> object:
    """

    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.pad_token = '</s>'
    return tokenizer

class SimplestLinearHead(nn.Module):
    def __init__(self, lm_output_size: int, num_classes: int):
        super(SimplestLinearHead, self).__init__()
        self.fc = nn.Linear(lm_output_size, num_classes)

    def forward(self, lm_hidden_states):
        pooled_output = torch.mean(lm_hidden_states, dim=1)
        logits = self.fc(pooled_output)
        return logits

@st.cache_resource  # 👈 Add the caching decorator
def load_checkpoint(hidden_size, num_classes) -> object:
    """

    """
    classifier = SimplestLinearHead(hidden_size, num_classes)
    #classifier = SimplestLinearHead(lm.config.hidden_size, num_classes).to(device)


    previous_checkpoint = torch.load(previous_checkpoint_file, map_location=torch.device("cpu"))
    classifier.load_state_dict(previous_checkpoint['classifier_state_dict'])

    return classifier


def predict(input: str, tokenizer: object, classifier:object, lm:object) -> (float, int):
    #classifier.eval()
    #lm.eval()
    #with torch.no_grad():
    print(f"the input is: {input}")
    tokenized_input = tokenizer.encode(input, return_tensors="pt")
    print(f"the tokenized input is: {tokenized_input}")
    lm_outputs = lm(tokenized_input)
    classifier_outputs = classifier(lm_outputs[0].float())

    # These classifier outputs are the logits
    # A call to torch.softmax(classifier_outputs) should do it!

    # to get probabilities for each label:
    #label_probs = torch.softmax(classifier_outputs)

    # to get the single most probable label:
    most_probable = classifier_outputs.argmax(dim=1)

    #print(f"LABEL PROBS ARE: {label_probs}")
    #print(f"MOST PROBABLE: {most_probable}")

    gradients = torch.autograd.grad(classifier_outputs, lm_outputs, grad_outputs=torch.ones_like(classifier_outputs), retain_graph=True)[0]

    return most_probable, gradients


