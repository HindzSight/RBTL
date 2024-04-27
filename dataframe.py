from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from wordcloud import STOPWORDS

sns.set(style='darkgrid', context='talk', palette='Dark2')

import praw

reddit = praw.Reddit(client_id='WDn0AqGI8iohUxu5BLcD0w',client_secret='_C0Zq9jevbdNmzjhHzhPJKHZ9C86IQ',user_agent='h1ndzs1ght')

# Extracting A subreddits hot posts Headlines for analysis

# subreddit_name = "politics"
# subreddit = reddit.subreddit(subreddit_name)

# posts = subreddit.hot(limit=50);

# p_list = []

# for post in posts:
#     post_text = post.title
#     p_list.append(post_text)
    
# print(p_list)
    

#Extracting Comments of a post using POST ID :: https://www.reddit.com/r/politics/comments/" Post ID : 184n3oo "/have_you_listened_lately_to_what_trump_is_saying/

# submission_id = "184n3oo"  # Replace with the actual post ID
# submission = reddit.submission(id=submission_id)

# for comment in submission.comments.list():
#     print(comment.body)

# p_set = set()

# for submissions in posts:
#     p_set.add(submissions.title)
#     display.clear_output()
#     print(p_set)

# print(posts)

headlines = set()
for submission in reddit.subreddit('politics').new(limit=200):
    headlines.add(submission.title)
    # display.clear_output()
    # print(len(headlines))
    # print(headlines)
    
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)

pprint(results[:3], width=100)

df = pd.DataFrame.from_records(results)
df.head()

#Printing the data that was recieved from the api along with marking the dataset 

df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
df.head()
# print(df.head())

    

df2 = df[['headline', 'label']]
df2.to_csv('reddit_headlines_labels.csv', mode='a', encoding='utf-8', index=False)

#Printing Positive and Negative WordCloud

print("Positive headlines:\n")
pprint(list(df[df['label'] == 1].headline)[:10], width=200)
pos_post = list(df[df['label'] == 1].headline)[:10]
stop_words = ["https", "co", "RT"] + list(STOPWORDS)
positive_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords = stop_words).generate(str(pos_post))
plt.figure()
plt.title("Positive Tweets - Wordcloud")
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

print("\nNegative headlines:\n")
pprint(list(df[df['label'] == -1].headline)[:10], width=200)
neg_post = list(df[df['label'] == -1].headline)[:10]
negative_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords = stop_words).generate(str(neg_post))
plt.figure()
plt.title("Negative Tweets - Wordcloud")
plt.imshow(negative_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


print(df.label.value_counts())

print(df.label.value_counts(normalize=True) * 100)

fig, ax = plt.subplots(figsize=(8, 8))

counts = df.label.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

plt.show()