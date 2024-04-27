# app.py
from flask import Flask, render_template, jsonify
from testModel import perform_sentiment_analysis
import praw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask_cors import CORS  # Added for cross-origin support
import threading
import time

sns.set(style='darkgrid', context='talk', palette='Dark2')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store sentiment analysis results
current_sia = None
lock = threading.Lock()

def fetch_and_analyze_headlines():
    reddit = praw.Reddit(client_id='WDn0AqGI8iohUxu5BLcD0w',
                         client_secret='_C0Zq9jevbdNmzjhHzhPJKHZ9C86IQ',
                         user_agent='h1ndzs1ght')

    headlines = set()
    for submission in reddit.subreddit('Politics').new(limit=10):
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

@app.route('/analyze')
def analyze():
    with lock:
        global current_sia
        current_sia = fetch_and_analyze_headlines()

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

            # Plotting the graph
            plt.figure(figsize=(8, 8))
            sns.barplot(x=sentiment_percentages.index, y=sentiment_percentages)
            plt.xlabel('Sentiment')
            plt.ylabel('Percentage')
            plt.title('Sentiment Analysis Results')

            # Save the plot to an image file
            plot_file_path = 'static/sentiment_analysis_plot.png'
            plt.savefig(plot_file_path)

            return render_template('liveupdateresult.html', plot_file_path=plot_file_path,
                                   sentiment_percentages=sentiment_percentages.to_dict(),
                                   positive_headlines=positive_headlines,
                                   negative_headlines=negative_headlines)

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

if __name__ == '__main__':
    # Start a thread to update sentiment analysis results in the background
    update_thread = threading.Thread(target=update_sentiment)
    update_thread.start()

    try:
        app.run(debug=True)
    except KeyboardInterrupt:
        # Handle keyboard interrupt to stop the background thread
        stop_event.set()
        update_thread.join()
        print("Flask app terminated.")