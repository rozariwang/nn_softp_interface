import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from datetime import datetime
import random_generator
import subprocess
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
#from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, BitsAndBytesConfig
from transformers import AutoModel, AutoTokenizer
from the_model import load_model, load_tokenizer, load_checkpoint, predict

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Main Page"

# Function to change the current page
def change_page(page_name):
    st.session_state['current_page'] = page_name

# Load credentials from Streamlit secrets
google_sheets_credentials = st.secrets["google_sheets_credentials"]

def init_google_sheets_connection():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(google_sheets_credentials, scope)
    client = gspread.authorize(creds)
    return client

def append_data_to_google_sheet(data, sheet_name, worksheet_index=0):
    client = init_google_sheets_connection()
    sheet = client.open(sheet_name).get_worksheet(worksheet_index)
    values_list = list(data.values())
    sheet.append_row(values_list)

def text_color_from_bg(bg_color):
    r, g, b, _ = bg_color
    brightness = r * 0.299 + g * 0.587 + b * 0.114  # Approximate brightness perception
    return "white" if brightness < 0.5 else "black"

def load_dataset(url, names=None):
    return pd.read_csv(url, names=names, header=None if names else 'infer')

def visualize_label_distribution(df, title):
    label_counts = df['label'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    ax.set_title(title)
    st.pyplot(fig)

liar_sizes = {
    "Train": 10240,  # example value
    "Test": 1267,  # example value
    "Validation": 1284  # example value
}

cofacts_sizes = {
    "Train": 5042,  # example value
    "Test": 631,  # example value
    "Validation": 629  # example value
}

def plot_pie_chart(sizes, title):
    fig, ax = plt.subplots()
    total = sum(sizes.values())
    # Custom autopct function to show both percentage and absolute value
    def autopct(pct):
        absolute = int(pct/100.*total)
        return "{:.1f}%\n({:d})".format(pct, absolute)
    ax.pie(sizes.values(), labels=sizes.keys(), autopct=autopct, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title(title)
    st.pyplot(fig)

# Page navigation buttons
st.sidebar.title("🚀 Navigation")
st.sidebar.button("Main Page", on_click=change_page, args=("Main Page",))
st.sidebar.button("Fact-Checking Links", on_click=change_page, args=("Fact-Checking Links",))
st.sidebar.button("Datasets", on_click=change_page, args=("Datasets",))
st.sidebar.button("Model Structure", on_click=change_page, args=("Model Structure",))

#########################
#### Main Page       ####
#########################
    
if 'current_page' in st.session_state and st.session_state['current_page'] == "Main Page":
    if 'classification_result' not in st.session_state:
        st.session_state['classification_result'] = ''
    if 'heatmap_html' not in st.session_state:
        st.session_state['heatmap_html'] = ''
    
    st.title("🚩 News Classifier 🚩")

    # Text input
    article_title = st.text_area("Article Title:", "Enter the article title here...")
    article_body = st.text_area("Article Body:", "Enter the article body here...")
    source_link = st.text_input('Enter the source link of the article:', '')
    threshold = st.number_input("Set the threshold for classification:", min_value=0.0, max_value=1.0, value=0.5)

    tokenizer = load_tokenizer()
    model = load_model()
    lm_hidden_size = model.hidden_size
    classifier = load_checkpoint(lm_hidden_size, 6)

    if st.button("Classify"):

        label_probs, most_prob = predict(input, classifier, tokenizer, model)
        print(f"MOST PROBABLE: {most_prob}")
        classification, surprisal_values, words = random_generator.generate_surprisal_values(article_body, threshold)
        st.session_state['classification_result'] = classification # Store the result in session state

        normalized_vals = np.interp(surprisal_values, (min(surprisal_values), max(surprisal_values)), (0, 1))
        colors = [cm.Reds(val) for val in normalized_vals]
        colors_hex = ["#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]
        heatmap_html = "".join([
            f'<span style="background-color: {colors_hex[i]}; color: {text_color_from_bg(colors[i])}; padding: 5px 10px; margin: 2px; border-radius: 5px; display: inline-block; min-width: 3em; text-align: center;">{word}</span>'
            for i, word in enumerate(words)
        ])
        st.session_state['heatmap_html'] = heatmap_html

     # Always display classification result and heatmap if available
    if st.session_state['classification_result']:
        #st.write(f"Classification: {st.session_state['classification_result']}")
        st.markdown(f"**Classification:** **{st.session_state['classification_result']}**", unsafe_allow_html=True)
        st.markdown(st.session_state['heatmap_html'], unsafe_allow_html=True)

     # Collect agreement and feedback
    classification_agreement = st.radio("Do you agree with the classification?", ["Yes", "No"], key='user_agreement')
    reason_for_disagreement = st.text_area("Please provide your reason for disagreement:", key='reason_for_disagreement')

    if st.button("Save"):
        data = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'title': article_title,
            'body': article_body,
            'link': source_link,
            'classification_result': st.session_state['classification_result'],
            'user_agreement': st.session_state['user_agreement'],
            'disagreement_reason': st.session_state['reason_for_disagreement'] if st.session_state['user_agreement'] == "No" else '',
        }
    
        # Specify your Google Sheet name here
        sheet_name = "Fake News HQ"
    
        try:
            append_data_to_google_sheet(data, sheet_name)
            st.success("Data saved to Google Sheet successfully!")
        except Exception as e:
            st.error(f"Failed to save data to Google Sheet: {e}")
        

#########################
#### Fact-Checking   ####
#########################
elif st.session_state['current_page'] == "Fact-Checking Links":
    st.title("Fact-Checking Links")
    # Displaying fact-checking links
    fact_checking_sites = {
        "FactCheck.org": "https://www.factcheck.org/",
        "PolitiFact": "https://www.politifact.com/",
        "Taiwan FactCheck Center": "https://tfc-taiwan.org.tw/",
        "Taiwan FactCheck Center (English)": "https://tfc-taiwan.org.tw/en",
        "Cofacts": "https://cofacts.tw/"
    }

    for site_name, site_url in fact_checking_sites.items():
        st.markdown(f"[{site_name}]({site_url})")

#########################
#### Dataset         ####
#########################


elif st.session_state['current_page'] == "Datasets":
    st.title("Datasets")
    
    # Sub-page selection
    dataset_choice = st.radio("Choose a dataset", ("LIAR", "Cofacts"))

    if dataset_choice == "LIAR":

        # Pie chart for LIAR dataset splits
        plot_pie_chart(liar_sizes, 'LIAR Dataset Split')
        
        # Column names for the LIAR dataset
        liar_columns = ['ID', 'label', 'Statement', 'Subject', 'Speaker', "Speaker's Job Title", 'State Info', 'Party Affiliation', 'Barely True Counts', 'False Counts', 'Half True Counts', 'Mostly True Counts', 'Pants on Fire Counts', 'Context']

        # LIAR Dataset URLs
        liar_datasets = {
            "LIAR Train": ('https://raw.githubusercontent.com/rozariwang/nn_softp_interface/main/liar_dataset/train.csv', liar_columns),
            "LIAR Test": ('https://raw.githubusercontent.com/rozariwang/nn_softp_interface/main/liar_dataset/test.csv', liar_columns),
            "LIAR Validation": ('https://raw.githubusercontent.com/rozariwang/nn_softp_interface/main/liar_dataset/valid.csv', liar_columns),
        }

        for name, (url, names) in liar_datasets.items():
            df = load_dataset(url, names=names)
            st.write(f"{name} Preview:")
            st.dataframe(df.head())
            visualize_label_distribution(df, f'{name} Label Distribution')

    elif dataset_choice == "Cofacts":

        # Pie chart for Cofacts dataset splits
        plot_pie_chart(cofacts_sizes, 'Cofacts Dataset Split')
        
        # Cofacts Dataset URLs
        cofacts_datasets = {
            "Cofacts Train": ('https://raw.githubusercontent.com/rozariwang/nn_softp_interface/main/cofacts_dataset/train.csv', None),
            "Cofacts Test": ('https://raw.githubusercontent.com/rozariwang/nn_softp_interface/main/cofacts_dataset/test.csv', None),
            "Cofacts Validation": ('https://raw.githubusercontent.com/rozariwang/nn_softp_interface/main/cofacts_dataset/validation.csv', None)
        }

        for name, (url, names) in cofacts_datasets.items():
            df = load_dataset(url, names=names)
            st.write(f"{name} Preview:")
            st.dataframe(df.head())
            visualize_label_distribution(df, f'{name} Label Distribution')



#########################
#### Model Structure ####
#########################
elif st.session_state['current_page'] == "Model Structure":
    st.title("Model Structure and Design")

    


