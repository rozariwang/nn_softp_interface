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

# Function to initialize Google Sheets API connection
def init_google_sheets_connection():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("path/to/your/google-credentials.json", scope)
    client = gspread.authorize(creds)
    return client

# Function to append data to the specified Google Sheet
def append_data_to_google_sheet(data, sheet_name, worksheet_index=0):
    client = init_google_sheets_connection()
    sheet = client.open(sheet_name).get_worksheet(worksheet_index)
    # Convert the data dictionary to a list of values and append to the Google Sheet
    values_list = list(data.values())
    sheet.append_row(values_list)


# Page navigation buttons
st.sidebar.title("🚀 Navigation")
st.sidebar.button("Main Page", on_click=change_page, args=("Main Page",))
st.sidebar.button("Fact-Checking Links", on_click=change_page, args=("Fact-Checking Links",))
st.sidebar.button("Dataset", on_click=change_page, args=("Datasets",))
st.sidebar.button("Model Structure", on_click=change_page, args=("Model Structure",))

#########################
#### Main Page       ####
#########################
if st.session_state['current_page'] == "Main Page":
    st.title("🚩 News Classifier 🚩")

    # Text input for the article title and body
    article_title = st.text_area("Article Title:", "Enter the article title here...")
    article_body = st.text_area("Article Body:", "Enter the article body here...")

    # Input for source link
    source_link = st.text_input('Enter the source link of the article:', '')

    # Numeric input for threshold
    threshold = st.number_input("Set the threshold for classification:", min_value=0.0, max_value=1.0, value=0.5)

    classification_agreement = st.radio(
        "Do you agree with the classification?",
        ["Yes", "No"],
    )

    # Feedback box
    reason_for_disagreement = st.text_area("Please provide your reason for disagreement:")

    if st.button("Classify"):
        classification, surprisal_values, words = random_generator.generate_surprisal_values(article_body, threshold)
        st.write(f"Classification: {classification}")


        # Normalize surprisal values for color mapping
        normalized_vals = np.interp(surprisal_values, (min(surprisal_values), max(surprisal_values)), (0, 1))
        colors = [cm.Reds(val) for val in normalized_vals]  # Change colormap to Reds

        # Function to determine text color based on background color
        def text_color_from_bg(bg_color):
            r, g, b, _ = bg_color
            brightness = r * 0.299 + g * 0.587 + b * 0.114  # Approximate brightness perception
            return "white" if brightness < 0.5 else "black"

        # Convert RGBA colors to HEX for HTML
        colors_hex = ["#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]

        # Generate HTML content with styled spans for each word
        html_content = "".join([
            f'<span style="background-color: {color}; color: {text_color_from_bg(rgba)}; padding: 5px 10px; margin: 2px; border-radius: 5px; display: inline-block; min-width: 3em; text-align: center;">{word}</span>'
            for word, color, rgba in zip(words, colors_hex, colors)
        ])


        # Display the custom heatmap in Streamlit
        st.markdown(html_content, unsafe_allow_html=True)


    if st.button("Save"):
        data = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'title': article_title,
            'body': article_body,
            'link': source_link,
            'classification_result': classification,
            'user_agreement': classification_agreement,
            'disagreement_reason': reason_for_disagreement if classification_agreement == "No" else '',
        }
    
        # Specify your Google Sheet name here
        sheet_name = "Your Google Sheet Name"
    
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
elif st.session_state['current_page'] == "Dataset":
    st.title("Dataset View")

    # Integrate with Awesome Table or simply display your dataset using Streamlit
    # For Awesome Table, you'll likely need to embed or link to it since direct integration
    # within Streamlit isn't straightforward without custom components.

#########################
#### Model Structure ####
#########################
elif st.session_state['current_page'] == "Model Structure":
    st.title("Model Structure and Design")

    


