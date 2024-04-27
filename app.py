# app.py
from flask import Flask, render_template
from testModel import perform_sentiment_analysis
import praw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')

app = Flask(__name__)

def fetch_and_analyze_headlines():
    reddit = praw.Reddit(client_id='WDn0AqGI8iohUxu5BLcD0w',
                         client_secret='_C0Zq9jevbdNmzjhHzhPJKHZ9C86IQ',
                         user_agent='h1ndzs1ght')

    headlines = set()
    for submission in reddit.subreddit('politics').new(limit=200):
        headlines.add(submission.title)

    sia = perform_sentiment_analysis(headlines)
    return sia

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    sia = fetch_and_analyze_headlines()

    # Create a DataFrame from the sentiment analysis results
    df = pd.DataFrame.from_records(sia)
    
    # Add a label to the DataFrame
    df['label'] = 0
    df.loc[df['compound'] > 0.2, 'label'] = 1
    df.loc[df['compound'] < -0.2, 'label'] = -1
    
    # Calculate percentage of sentiments
    sentiment_percentages = df['label'].value_counts(normalize=True) * 100
    
    # Plotting the graph
    plt.figure(figsize=(8, 8))
    sns.barplot(x=sentiment_percentages.index, y=sentiment_percentages)
    plt.xlabel('Sentiment')
    plt.ylabel('Percentage')
    plt.title('Sentiment Analysis Results')
    plt.xticks(ticks=[0, 1, 2], labels=['Negative', 'Neutral', 'Positive'])
    
    # Save the plot to an image file
    plot_file_path = 'static/sentiment_analysis_plot.png'
    plt.savefig(plot_file_path)

    return render_template('result.html', plot_file_path=plot_file_path)

if __name__ == '__main__':
    app.run(debug=True)