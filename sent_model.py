# sent_model.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def perform_sentiment_analysis_sent(texts):
    sia = SentimentIntensityAnalyzer()

    results = []

    for text in texts:
        pol_score = sia.polarity_scores(text)
        pol_score['headline'] = text
        results.append(pol_score)

    return results
