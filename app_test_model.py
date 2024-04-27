# app.py
from flask import Flask, render_template, jsonify, request, send_from_directory
from testModel import perform_sentiment_analysis
import praw
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from flask_cors import CORS  # Added for cross-origin support
from arin_model import perform_sentiment_analysis_sent
import threading
import time
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure matplotlib to use the Agg backend
matplotlib.use('Agg')

sns.set(style='darkgrid', context='talk', palette='Dark2')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store sentiment analysis results and cleaned text
current_sia = None
current_cleaned = None
lock = threading.Lock()

# Function to perform emotion analysis
def analyze_emotions(text):
    global current_cleaned
    lower_case = text.lower()
    cleaned = lower_case.translate(str.maketrans('', '', string.punctuation))
    current_cleaned = cleaned

    tokenized_words = word_tokenize(cleaned, "english")

    result_words = []
    for words in tokenized_words:
        if words not in stopwords.words('english'):
            result_words.append(words)

    emotions = []
    with open('emotional_damage.txt', 'r') as file:
        for line in file:
            clean_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
            words, emotion = clean_line.split(':')
            if words in result_words:
                emotions.append(emotion)

    emotion_counter = Counter(emotions)
    return emotion_counter

def fetch_and_analyze_headlines(subprovided):
    reddit = praw.Reddit(client_id='WDn0AqGI8iohUxu5BLcD0w',
                         client_secret='_C0Zq9jevbdNmzjhHzhPJKHZ9C86IQ',
                         user_agent='h1ndzs1ght')

    headlines = set()
    for submission in reddit.subreddit(subprovided).new(limit=100):
        headlines.add(submission.title)

    sia = perform_sentiment_analysis(headlines)
    return sia

def update_sentiment():
    global current_sia
    while True:
        with lock:
            current_sia = fetch_and_analyze_headlines()
        time.sleep(10)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    input_text = data.get('text', '')
    analysis_type = data.get('type', 'sentence')

    if analysis_type == 'sentence':
        # Perform sentiment analysis on the input sentence
        sia = perform_sentiment_analysis_sent([input_text])
    elif analysis_type == 'community':
        # Perform sentiment analysis on headlines from a specified subreddit
        subreddit_name = data.get('subreddit', 'Politics')  # Default subreddit is 'Politics'
        sia = fetch_and_analyze_headlines(subreddit_name)
    else:
        return jsonify({'error': 'Invalid analysis type'})

    with lock:
        global current_sia
        current_sia = sia

        if current_sia is not None:
            # Create a DataFrame from the sentiment analysis results
            df = pd.DataFrame.from_records(current_sia)

            # Add a label to the DataFrame
            df['label'] = 0
            df.loc[df['compound'] > 0.2, 'label'] = 1
            df.loc[df['compound'] < -0.2, 'label'] = -1

            # Calculate percentage of sentiments
            sentiment_percentages = df['label'].value_counts(normalize=True) * 100

            # Extract positive and negative headlines
            positive_headlines = list(df[df['label'] == 1]['headline'])
            negative_headlines = list(df[df['label'] == -1]['headline'])

            # Plotting the sentiment analysis graph
            plt.figure(figsize=(8, 8))
            sns.barplot(x=sentiment_percentages.index, y=sentiment_percentages)
            plt.xlabel('Sentiment')
            plt.ylabel('Percentage')
            plt.title('Sentiment Analysis Results')

            # Save the sentiment plot to an image file
            plot_file_path = 'static/sentiment_analysis_plot.png'
            plt.savefig(plot_file_path)

            # Perform emotion analysis on the cleaned text
            emotions = analyze_emotions(input_text)

            # Plotting the emotion analysis graph
            plt.figure(figsize=(8, 8))
            plt.bar(emotions.keys(), emotions.values())
            plt.xlabel('Emotion')
            plt.ylabel('Count')
            plt.title('Emotion Analysis Results')

            # Save the emotion plot to an image file
            emotion_plot_file_path = 'static/emotion_analysis_plot.png'
            plt.savefig(emotion_plot_file_path)

            return render_template('liveupdateresult.html', plot_file_path=plot_file_path,
                                   sentiment_percentages=sentiment_percentages.to_dict(),
                                   positive_headlines=positive_headlines,
                                   negative_headlines=negative_headlines,
                                   emotion_plot_file_path=emotion_plot_file_path,
                                   emotions=emotions)

    return render_template('liveupdateresult.html')

@app.route('/get_sentiment')
def get_sentiment():
    with lock:
        global current_sia
        if current_sia is None:
            return jsonify({'sentiment_percentages': {'Positive': 0, 'Negative': 0},
                            'positive_headlines': [],
                            'negative_headlines': []})

        # Create a DataFrame from the sentiment analysis results
        df = pd.DataFrame.from_records(current_sia)

        # Add a label to the DataFrame
        df['label'] = 0
        df.loc[df['compound'] > 0.2, 'label'] = 1
        df.loc[df['compound'] < -0.2, 'label'] = -1

        # Calculate percentage of sentiments
        sentiment_percentages = df['label'].value_counts(normalize=True) * 100

        # Extract positive and negative headlines
        positive_headlines = list(df[df['label'] == 1]['headline'])
        negative_headlines = list(df[df['label'] == -1]['headline'])

        return jsonify({'sentiment_percentages': sentiment_percentages.to_dict(),
                        'positive_headlines': positive_headlines,
                        'negative_headlines': negative_headlines})


@app.route('/result')
def result():
    with lock:
        global current_sia, current_cleaned
        if current_sia is None:
            return render_template('result.html', sentiment_percentages={'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Polarity': 0})

        # Create a DataFrame from the sentiment analysis results
        df = pd.DataFrame.from_records(current_sia)

        # Add a label to the DataFrame
        df['label'] = 0
        df.loc[df['compound'] > 0.2, 'label'] = 1
        df.loc[df['compound'] < -0.2, 'label'] = -1

        # Calculate percentage of sentiments
        sentiment_percentages = df['label'].value_counts(normalize=True) * 100

        # Extract positive and negative headlines
        positive_headlines = list(df[df['label'] == 1]['headline'])
        negative_headlines = list(df[df['label'] == -1]['headline'])

        # Plotting the sentiment analysis graph
        plt.figure(figsize=(8, 8))
        sns.barplot(x=sentiment_percentages.index, y=sentiment_percentages)
        plt.xlabel('Sentiment')
        plt.ylabel('Percentage')
        plt.title('Sentiment Analysis Results')

        # Save the sentiment plot to an image file
        plot_file_path = 'static/sentiment_analysis_plot.png'
        plt.savefig(plot_file_path)

        # Perform emotion analysis on the cleaned text
        emotions = analyze_emotions(current_cleaned)  # Call the function with the cleaned text

        # Plotting the emotion analysis graph
        plt.figure(figsize=(8, 8))
        plt.bar(emotions.keys(), emotions.values())
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.title('Emotion Analysis Results')

        # Save the emotion plot to an image file
        emotion_plot_file_path = 'static/emotion_analysis_plot.png'
        plt.savefig(emotion_plot_file_path)

        return render_template('result.html', sentiment_percentages={
            'Positive': sentiment_percentages.get(1, 0),
            'Negative': sentiment_percentages.get(-1, 0),
            'Neutral': sentiment_percentages.get(0, 0),
            'Polarity': sentiment_percentages.get(1, 0) - sentiment_percentages.get(-1, 0),
        }, plot_file_path=plot_file_path,
            positive_headlines=positive_headlines,
            negative_headlines=negative_headlines,
            emotion_plot_file_path=emotion_plot_file_path,
            emotions=emotions)

@app.route('/liveupdateresult')
def liveupdateresult():
    return render_template('liveupdateresult.html')


if __name__ == '__main__':
    # Start a thread to update sentiment analysis results in the background
    update_thread = threading.Thread(target=update_sentiment)
    update_thread.start()

    try:
        app.run(debug=True)
    except KeyboardInterrupt:
        # Handle keyboard interrupt to stop the background thread
        update_thread.join()
        print("Flask app terminated.")
