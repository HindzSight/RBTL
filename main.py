import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plott
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



text = open('read.txt',encoding='utf-8').read()
# print(text)

lower_case = text.lower()    #lower downed the cases here
# print(lower_case)

# print(string.punctuation)

cleaned = lower_case.translate(str.maketrans('','',string.punctuation))




# tokenized_words = cleaned.split()
tokenized_words = word_tokenize(cleaned,"english")

# print(cleaned)

# print(tokenized_words)


# stop_word = ["a", "about", "above", "after", "again", "against", "ain't", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
#              "be", "because", "been", "before", "being", "below", "between", "both", "but", "by","can't", "cannot", "could", "couldn't",
#              "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during","each","few", "for", "from", "further",
#              "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
#              "here's", "hers", "herself", "him", "himself", "his", "how", "how's","i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
#              "is", "isn't", "it", "it's", "its", "itself","let's","me", "more", "most", "mustn't", "my", "myself","no", "nor", "not",
#              "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
#              "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such","than", "that",
#              "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
#              "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too","under", "until", "up","very",
#              "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
#              "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
#              "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves","he", "him", "his", "himself", "she", "her", "hers",
#              "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
#              "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
#              "do", "does", "did", "doing"]

result_words = [] #empty list

#now we'll use loop as there can be thousands and thousands of words

for words in tokenized_words:
    if words not in stopwords.words('english'):
        result_words.append(words)


# print(result_words)

Emotions = []  #list of emotions
with open('emotional_damage.txt','r') as file:
    for line in file:
       clean_line = line.replace('\n','').replace(',','').replace("'",'').strip()   #clearing the punctuations
      #print(clean_line)
       words, emotion = clean_line.split(':')  # so i am pointing the words with their emotions
       #print("Word :" + word + "         " + "Emotions :" + emotion)
       if words in result_words:
           Emotions.append(emotion)
# print(Emotions)
w = Counter(Emotions)
# print(w)

def sa(st):
    score = SentimentIntensityAnalyzer().polarity_scores(st)
    neg = score['neg']
    pos = score['pos']
    # print(score)
    # if neg > pos:
    #     print("Negative")
    # else:
    #     print("positive")

sa(cleaned)

fig, ax1 = plott.subplots()

ax1.bar(w.keys(),w.values())
fig.autofmt_xdate()
plott.savefig('graph.png')
plott.show()

#In the above three string sapce what does it takes

#str 1: Specifies the list of characters that needs to be replaced
#str 2: Specifies the list of characters with which character needs to be replaced
#str 3: Specifies the list of characters that needs to be deleted

#str 1:-> 'abc'
#str 2:-> 'gfh'

#Returns : Returns the translation table which specifies the conversions that can be used by