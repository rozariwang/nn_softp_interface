import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from datetime import datetime
import random_generator
import subprocess
import os

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Main Page"

# Function to change the current page
def change_page(page_name):
    st.session_state['current_page'] = page_name

# Page navigation buttons
st.sidebar.title("Navigation")
st.sidebar.button("Main Page", on_click=change_page, args=("Main Page",))
st.sidebar.button("Fact-Checking Links", on_click=change_page, args=("Fact-Checking Links",))
st.sidebar.button("Dataset", on_click=change_page, args=("Dataset",))
st.sidebar.button("Model Structure", on_click=change_page, args=("Model Structure",))

#########################
#### Main Page       ####
#########################
if st.session_state['current_page'] == "Main Page":
    st.title("News Classifier with Surprisal Values")

   # Text input for the article title and body
   article_title = st.text_area("Article Title:", "Enter the article title here...")
   article_body = st.text_area("Article Body:", "Enter the article body here...")

   # Input for source link
   source_link = st.text_input('Enter the source link of the article:', '')

   # Numeric input for threshold
   threshold = st.number_input("Set the threshold for classification:", min_value=0.0, max_value=1.0, value=0.5)

   # Initialize session state for classification agreement and reason for disagreement
   if 'classification_agreement' not in st.session_state:
       st.session_state['classification_agreement'] = None
   if 'reason_for_disagreement' not in st.session_state:
       st.session_state['reason_for_disagreement'] = ''
    
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

       # Define a key for the select box to easily access its value from st.session_state
       classification_agreement = st.radio(
           "Do you agree with the classification?",
           ["Yes", "No"],
       )

       # A button to confirm the choice
       if st.button('Confirm'):
           # Use session_state to remember the user's choice even after rerun
           st.session_state['confirmed_agreement'] = classification_agreement

           # Once confirmed, display the feedback box if they disagreed
           if classification_agreement == "No":
               st.session_state['show_feedback_box'] = True
           else:
               st.session_state['show_feedback_box'] = False
               # Optionally, handle the "Yes" case or reset state as necessary

       # Conditional display based on session_state, to persist across reruns
       if st.session_state.get('show_feedback_box', False):
           reason_for_disagreement = st.text_area("Please provide your reason for disagreement:")
           st.session_state['reason_for_disagreement'] = reason_for_disagreement
   
           # Saving data to CSV
           data = {
               'date': [datetime.now()],
               'title': [article_title],
               'body': [article_body],
               'link': [source_link],
               'classification_result': [classification],
               'user_agreement': [classification_agreement],
               'disagreement_reason': [reason_for_disagreement if classification_agreement == "No" else ''],
           }
           df = pd.DataFrame(data)
           # Append the data to 'data.csv', creating if doesn't exist
           df.to_csv('data.csv', mode='a', header=not pd.read_csv('data.csv').empty, index=False)

    
   repo_path = '/the/correct/full/path/to/your/git/repository'  # Update this

   if st.button("Save"):
       try:
           # Change to the repository directory
           os.chdir(repo_path)
        
           # Attempt to add, commit, and push changes
           subprocess.run(["git", "add", "data.csv"], check=True)
           commit_result = subprocess.run(["git", "commit", "-m", "Update data.csv"], capture_output=True, text=True)
        
           # Only attempt to push if the commit was successful
           if commit_result.returncode == 0:
               push_result = subprocess.run(["git", "push"], check=True, capture_output=True, text=True)
               st.success("Changes saved and pushed to GitHub successfully!")
           else:
               st.error(f"Failed to commit changes. {commit_result.stderr}")
            
       except subprocess.CalledProcessError as e:
           st.error(f"An error occurred: {e}")
        

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

    


