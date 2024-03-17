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


    if st.button("Classify"):
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
    
    liar_train_url = 'https://github.com/rozariwang/nn_softp_interface/tree/e8d820e9d91a8acd61bb1191f2d86ccef196db93/liar_dataset/train.csv'
    #cofacts_train_url = 'URL_TO_COFCTS_TRAIN_DATASET'  # Note: GitHub doesn't support Parquet directly; consider converting to CSV or hosting elsewhere

    liar_df = pd.read_csv(liar_train_url, sep='\t', header=None)
    liar_df.columns = ['ID', 'Label', 'Statement', 'Subject', 'Speaker', 'Speaker\'s Job Title', 'State Info', 'Party Affiliation', 'Barely True Counts', 'False Counts', 'Half True Counts', 'Mostly True Counts', 'Pants on Fire Counts', 'Context']
   
    st.write("LIAR Dataset Preview:")
    st.dataframe(liar_df.head())

    # Function to visualize label distribution
    def visualize_label_distribution(df, title):
        label_counts = df['Label'].value_counts()
        fig, ax = plt.subplots()
        label_counts.plot(kind='bar', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    # Visualize label distribution for LIAR dataset
    visualize_label_distribution(liar_df, 'LIAR Dataset Label Distribution')

    
   

    # Integrate with Awesome Table or simply display your dataset using Streamlit
    # For Awesome Table, you'll likely need to embed or link to it since direct integration
    # within Streamlit isn't straightforward without custom components.

#########################
#### Model Structure ####
#########################
elif st.session_state['current_page'] == "Model Structure":
    st.title("Model Structure and Design")

    


