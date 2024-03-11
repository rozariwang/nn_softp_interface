import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from datetime import datetime
import random_generator

#########################
#### Visual Elements ####
#########################

st.title("News Classifier with Surprisal Values")

# Text input for the article title and body
article_title = st.text_area("Article Title:", "Enter the article title here...")
article_body = st.text_area("Article Body:", "Enter the article body here...")

# Input for source link
source_link = st.text_input('Enter the source link of the article:', '')

# Numeric input for threshold
threshold = st.number_input("Set the threshold for classification:", min_value=0.0, max_value=1.0, value=0.5)

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

    # User feedback on classification result
    classification_agreement = st.selectbox("Do you agree with the classification?", ["Yes", "No"])

    reason_for_disagreement = ""
    if classification_agreement == "No":
        reason_for_disagreement = st.text_area("Please provide your reason for disagreement:")


#############################
##### Saving User Input #####
#############################

# Saving data to CSV
    data = {
        'date': [datetime.now()],
        'title': [article_title],
        'body': [article_body],
        'link': [source_link],
        'classification_result': [classification],
        'user_agreement': [classification_agreement],
        'disagreement_reason': [reason_for_disagreement],
    }
    df = pd.DataFrame(data)
    # Appending the data to 'data.csv', creating if doesn't exist
    df.to_csv('data.csv', mode='a', header=not pd.read_csv('data.csv').empty, index=False)

'''
#######################################
#### Extract Input Text from Links ####
#######################################

# Initialize the Tweets scraper
scraper = Nitter()

def extract_tweet(tweet_id):
    """
    Extracts a single tweet's text given its ID using Nitter scraper.
    """
    try:
        # Attempt to fetch the tweet by its ID
        tweets_data = scraper.get_tweets(tweet_id, mode='id', number=1)

        if tweets_data and 'tweets' in tweets_data and len(tweets_data['tweets']) > 0:
            # Extract 'text' from the first tweet in the list (It still returns a list since it is the way Niter works)
            tweet_text = tweets_data['tweets'][0]['text']
            return tweet_text
        else:
            return "Tweet could not be retrieved or does not exist."
    except Exception as e:
        return f"An error occurred: {e}"


def extract_news_article(url):
    # Send a GET request to the article's URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Attempt to find the article's title. Some websites use <h1> or other tags for the main title.
        if soup.find('title'):
            title = soup.find('title').text
        elif soup.find('h1'):
            title = soup.find('h1').text
        else:
            title = 'No title found'

        # If there's no specific container, fall back to extracting from all paragraph tags
        article_body = soup.find_all('p')
        
        # Join all the text from the paragraphs, and clean it
        article_text = ' '.join(p.get_text(separator=' ', strip=True) for p in article_body)
        
        return title, article_text
    else:
        return 'Failed to retrieve the article', 'The request to the article URL did not succeed.'


# Check if a URL is provided and try to extract content
if input_url:
    if 'twitter.com' in input_url:
        # Extract tweet ID from the URL and its content
        tweet_id = re.findall(r'twitter\.com/.+/status/(\d+)', input_url)
        if tweet_id:
            try:
                # Assuming `extract_tweet` function and `twitter_api` are defined and set up
                tweet_text = extract_tweet(tweet_id)  
                user_input = tweet_text
                st.text_area("Extracted tweet text:", tweet_text, height=150)
            except Exception as e:
                st.error(f"Error extracting tweet: {e}")
        else:
            st.error('Could not extract tweet ID from the URL.')
    else:
        # Extract news article content
        try:
            title, text = extract_news_article(input_url) 
            user_input = text
            st.text_area("Extracted article text:", text, height=150)
        except Exception as e:
            st.error(f"Error extracting article content: {e}")
       

# Numeric input for threshold
threshold = st.number_input("Set the threshold for classification:", min_value=0.0, max_value=1.0, value=0.5)
'''

