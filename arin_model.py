import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def perform_sentiment_analysis_sent(sentences):
    results = []

    for sentence in sentences:
        lower_case = sentence.lower()
        cleaned = lower_case.translate(str.maketrans('', '', string.punctuation))
        tokenized_words = word_tokenize(cleaned, "english")

        result_words = [word for word in tokenized_words if word not in stopwords.words('english')]

        Emotions = []
        with open('emotional_damage.txt', 'r') as file:
            for line in file:
                clean_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
                words, emotion = clean_line.split(':')
                if words in result_words:
                    Emotions.append(emotion)

        w = Counter(Emotions)

        # Perform VADER sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(cleaned)

        # Create a dictionary representing the sentiment analysis result for the current sentence
        result = {
            'headline': sentence,
            'compound': score['compound'],
            'neg': score['neg'],
            'neu': score['neu'],
            'pos': score['pos'],
            'emotions': dict(w)
        }

        results.append(result)

    return results
