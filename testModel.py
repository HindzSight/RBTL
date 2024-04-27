# your_sentiment_analysis_module.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def perform_sentiment_analysis(texts):
    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    results = []

    for line in texts:
        pol_score = sia.polarity_scores(line)
        pol_score['headline'] = line
        results.append(pol_score)

    return results
