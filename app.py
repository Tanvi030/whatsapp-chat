import streamlit as st
import pandas as pd
from preprocessor import preprocess_chat, sort_chat
from helper import analyze_activity, plot_heatmap, plot_timeline, plot_wordcloud, analyze_sentiment, \
    train_topic_classifier, predict_topics, get_stats
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Set page configuration for a wider layout and custom theme
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide", page_icon="📱")

# Custom CSS for improved aesthetics
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #25D366; color: white; border-radius: 8px;}
    .stSelectbox, .stFileUploader {background-color: white; border-radius: 8px; padding: 10px;}
    .sidebar .sidebar-content {background-color: #e6f3e6;}
    h1, h2, h3 {color: #075E54; font-family: 'Arial', sans-serif;}
    .stDataFrame {border: 1px solid #25D366; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# Sidebar with navigation and app info
st.sidebar.title("WhatsApp Chat Analyzer 📱")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/733/733585.png", width=100, caption="Chat Analysis")
st.sidebar.markdown("Upload a WhatsApp chat file (.txt) to analyze messages, sentiments, topics, and more!")
st.sidebar.markdown("**Features:**")
st.sidebar.markdown("- 📊 Chat Statistics")
st.sidebar.markdown("- 🌡️ Activity Heatmap")
st.sidebar.markdown("- 📈 Message Timeline")
st.sidebar.markdown("- ☁️ Word Cloud")
st.sidebar.markdown("- 😊 Sentiment Analysis")
st.sidebar.markdown("- 🏷️ Topic Prediction")
st.sidebar.markdown("- 📏 Message Length Analysis")
st.sidebar.markdown("- 😄 Emoji Usage Prediction")

# Main title and description
st.title("WhatsApp Chat Analyzer")
st.markdown("Explore your WhatsApp conversations with interactive analytics and insights! Upload a chat file to get started.")

# File uploader
uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type="txt", help="Export your WhatsApp chat as a .txt file and upload it here.")

if uploaded_file:
    # Save uploaded file
    with open("chat.txt", "wb") as f:
        f.write(uploaded_file.read())

    # Preprocess chat
    df = preprocess_chat("chat.txt")
    if df is None or df.empty:
        st.error("Failed to process chat file. Please check the file format and ensure it contains valid messages.")
    else:
        st.success("Chat loaded successfully! 🎉")

        # Sorting options
        st.subheader("Sort Messages")
        col1, col2 = st.columns([2, 1])
        with col1:
            sort_by = st.selectbox("Sort by", ['Timestamp', 'Sender', 'Message'], help="Choose a column to sort the chat")
        with col2:
            ascending = st.checkbox("Ascending", value=True)
        df = sort_chat(df, sort_by, ascending)
        if df is None or df.empty:
            st.error("Error sorting chat. No valid data available.")
        else:
            # Display sorted chat
            st.subheader("Chat Preview")
            st.dataframe(df[['Timestamp', 'Sender', 'Message']].head(100), use_container_width=True)

            # Basic stats
            stats = get_stats(df)
            st.subheader("Chat Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Messages", stats['total_messages'])
            col2.metric("Total Senders", stats['total_senders'])
            col3.metric("Total Emojis", stats['total_emojis'])
            col4.write("**Top 5 Senders:**")
            col4.write(stats['top_senders'].to_frame().reset_index().to_markdown(index=False))

            # Activity analysis
            st.subheader("Activity Analysis")
            activity = analyze_activity(df)
            if activity is not None:
                plot_heatmap(activity)
                st.image("heatmap.png", caption="Chat Activity Heatmap", use_container_width=True)

            # Timeline
            st.subheader("Message Timeline")
            plot_timeline(df)
            st.image("timeline.png", caption="Messages Over Time", use_container_width=True)

            # Word Cloud
            st.subheader("Word Cloud")
            plot_wordcloud(df)
            st.image("wordcloud.png", caption="Word Cloud of Messages", use_container_width=True)

            # Sentiment Analysis
            st.subheader("Sentiment Analysis")
            df = analyze_sentiment(df)
            if df is not None:
                st.dataframe(df[['Timestamp', 'Sender', 'Message', 'Sentiment']].head(50), use_container_width=True)
                st.write("**Sentiment Distribution:**")
                st.bar_chart(df['Sentiment'].value_counts(bins=10))

            # Topic Prediction
            st.subheader("Topic Prediction")
            pipeline = train_topic_classifier()
            df = predict_topics(df, pipeline)
            if df is not None:
                st.dataframe(df[['Timestamp', 'Sender', 'Message', 'Topic']].head(50), use_container_width=True)
                st.write("**Topic Distribution:**")
                st.bar_chart(df['Topic'].value_counts())

            # Message Length Analysis
            st.subheader("Message Length Analysis")
            df['Length'] = df['Message'].apply(len)
            bins = [0, 50, 150, float('inf')]
            labels = ['Short', 'Medium', 'Long']
            df['Length_Category'] = pd.cut(df['Length'], bins=bins, labels=labels, include_lowest=True)
            st.dataframe(df[['Timestamp', 'Sender', 'Message', 'Length_Category']].head(50), use_container_width=True)
            st.write("**Message Length Distribution:**")
            st.bar_chart(df['Length_Category'].value_counts())

            # Emoji Usage Prediction
            st.subheader("Emoji Usage Prediction")
            df['Has_Emoji'] = df['Emojis'].apply(lambda x: 1 if x else 0)
            emoji_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', LogisticRegression())
            ])
            emoji_pipeline.fit(df['Message'], df['Has_Emoji'])
            df['Emoji_Prediction'] = emoji_pipeline.predict(df['Message'])
            st.dataframe(df[['Timestamp', 'Sender', 'Message', 'Has_Emoji', 'Emoji_Prediction']].head(50), use_container_width=True)
            st.write("**Emoji Usage Distribution:**")
            st.bar_chart(df['Has_Emoji'].value_counts().rename({0: 'No Emoji', 1: 'Has Emoji'}))
else:
    st.info("Please upload a WhatsApp chat file to begin analysis. 📤")
