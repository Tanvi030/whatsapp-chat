import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
import emoji
import numpy as np

plt.style.use('seaborn-v0_8')  # Updated style

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)


def analyze_activity(df):
    """Analyze chat activity patterns."""
    if df is None or df.empty:
        print("Error: Empty DataFrame in analyze_activity.")
        return None
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day_name()
    activity = df.groupby(['Hour', 'Day'])['Message'].count().reset_index()
    return activity


def plot_heatmap(activity):
    """Plot heatmap of chat activity."""
    if activity is None or activity.empty:
        print("Error: No activity data to plot heatmap.")
        plt.figure(figsize=(15, 8))
        plt.title('Chat Activity Heatmap (No Data)')
        plt.text(0.5, 0.5, 'No activity data available', horizontalalignment='center', verticalalignment='center')
        plt.savefig('heatmap.png')
        plt.close()
        return
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    activity_pivot = activity.pivot(index='Day', columns='Hour', values='Message').reindex(day_order)
    if activity_pivot.empty:
        print("Error: Pivot table is empty.")
        plt.figure(figsize=(15, 8))
        plt.title('Chat Activity Heatmap (No Data)')
        plt.text(0.5, 0.5, 'No activity data available', horizontalalignment='center', verticalalignment='center')
        plt.savefig('heatmap.png')
        plt.close()
        return
    plt.figure(figsize=(15, 8))
    sns.heatmap(activity_pivot, cmap='Blues', annot=True, fmt='.0f')
    plt.title('Chat Activity Heatmap')
    plt.savefig('heatmap.png')
    plt.close()


def plot_timeline(df):
    """Plot message frequency over time."""
    if df is None or df.empty:
        print("Error: Empty DataFrame in plot_timeline.")
        plt.figure(figsize=(15, 5))
        plt.title('Messages Over Time (No Data)')
        plt.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
        plt.savefig('timeline.png')
        plt.close()
        return
    timeline = df.groupby(df['Timestamp'].dt.date)['Message'].count()
    plt.figure(figsize=(15, 5))
    timeline.plot(kind='line')
    plt.title('Messages Over Time')
    plt.xlabel('Date')
    plt.ylabel('Message Count')
    plt.savefig('timeline.png')
    plt.close()


def plot_wordcloud(df):
    """Generate word cloud from messages."""
    if df is None or df.empty:
        print("Error: Empty DataFrame in plot_wordcloud.")
        plt.figure(figsize=(10, 5))
        plt.title('Word Cloud (No Data)')
        plt.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
        plt.savefig('wordcloud.png')
        plt.close()
        return
    text = ' '.join(df['Message'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('wordcloud.png')
    plt.close()


def analyze_sentiment(df):
    """Perform sentiment analysis on messages."""
    if df is None or df.empty:
        print("Error: Empty DataFrame in analyze_sentiment.")
        return None
    sid = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Message'].apply(lambda x: sid.polarity_scores(x)['compound'])
    return df


def train_topic_classifier():
    """Train a simple topic classifier (mock example)."""
    data = {
        'text': [
            'What is machine learning?', 'Supervised learning is great', 'Deep learning models',
            'Big data processing with Hadoop', 'Spark is fast for big data', 'Big data analytics',
            'ReactJS components are reusable', 'State management in React', 'React hooks tutorial'
        ],
        'topic': ['ML', 'ML', 'ML', 'Big Data', 'Big Data', 'Big Data', 'ReactJS', 'ReactJS', 'ReactJS']
    }
    df_train = pd.DataFrame(data)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    pipeline.fit(df_train['text'], df_train['topic'])
    return pipeline


def predict_topics(df, pipeline):
    """Predict topics for chat messages."""
    if df is None or df.empty:
        print("Error: Empty DataFrame in predict_topics.")
        return None
    df['Topic'] = pipeline.predict(df['Message'])
    return df


def get_stats(df):
    """Calculate basic chat statistics."""
    if df is None or df.empty:
        print("Error: Empty DataFrame in get_stats.")
        return {
            'total_messages': 0,
            'total_senders': 0,
            'total_emojis': 0,
            'top_senders': pd.Series()
        }
    total_messages = len(df)
    total_senders = df['Sender'].nunique()
    total_emojis = df['Emojis'].apply(lambda x: len(x)).sum()
    top_senders = df['Sender'].value_counts().head(5)
    return {
        'total_messages': total_messages,
        'total_senders': total_senders,
        'total_emojis': total_emojis,
        'top_senders': top_senders
    }
