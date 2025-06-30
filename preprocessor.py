import pandas as pd
import re
import emoji


def preprocess_chat(file_path):
    """Parse and preprocess WhatsApp chat data."""
    try:
        # Read chat file
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            print("Error: Chat file is empty.")
            return None

        data = []
        # Updated pattern to capture common WhatsApp formats
        pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s[AP]M)?)\s-\s([^:]+):\s(.+)'

        for line in lines:
            match = re.match(pattern, line)
            if match:
                timestamp, sender, message = match.groups()
                # Extract emojis
                emojis = ''.join(c for c in message if c in emoji.EMOJI_DATA)
                # Clean message
                message = re.sub(r'http\S+', '', message)  # Remove URLs
                message = message.strip()
                if message:  # Only append if message is non-empty
                    data.append([timestamp, sender, message, emojis])
            else:
                print(f"Skipping line (no match): {line.strip()}")

        if not data:
            print("Error: No valid chat messages found. Check file format.")
            return None

        # Create DataFrame
        df = pd.DataFrame(data, columns=['Timestamp', 'Sender', 'Message', 'Emojis'])

        # Handle missing or empty messages
        df = df.dropna(subset=['Message'])
        df = df[df['Message'] != '']

        if df.empty:
            print("Error: DataFrame is empty after preprocessing.")
            return None

        # Convert Timestamp to datetime with multiple format attempts
        date_formats = [
            '%m/%d/%y, %I:%M %p',  # e.g., 6/4/25, 12:22 AM
            '%d/%m/%y, %H:%M',  # e.g., 04/06/25, 00:22
            '%d/%m/%Y, %H:%M',  # e.g., 04/06/2025, 00:22
            '%m/%d/%Y, %I:%M %p',  # e.g., 06/04/2025, 12:22 AM
        ]

        for fmt in date_formats:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=fmt, errors='raise')
                break
            except ValueError:
                continue
        else:
            print("Error: Could not parse timestamps. Please check the date format in the chat file.")
            return None

        print(f"Preprocessed {len(df)} messages successfully.")
        return df
    except Exception as e:
        print(f"Error preprocessing chat: {e}")
        return None


def sort_chat(df, sort_by='Timestamp', ascending=True):
    """Sort chat DataFrame by specified column."""
    try:
        if df is None or df.empty:
            print("Error: Cannot sort empty DataFrame.")
            return None
        if sort_by in df.columns:
            return df.sort_values(by=sort_by, ascending=ascending)
        else:
            print(f"Invalid sort column: {sort_by}")
            return df
    except Exception as e:
        print(f"Error sorting chat: {e}")
        return df
