import pandas as pd
import string
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
import os

def load_data(file_path, num_rows=5000):
    """
    Load data from the provided CSV file path and return the first num_rows rows.

    Args:
        file_path (str): The file path to the CSV file.
        num_rows (int): The number of rows to return from the dataset.

    Returns:
        pandas.DataFrame: Loaded DataFrame with a limited number of rows.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    
    df = pd.read_csv(file_path)
    
    # Return only the first num_rows rows
    return df.head(num_rows)


def preprocess_text(text):
    """
    Preprocess the text by normalizing, tokenizing, and removing stop words.
    
    Args:
        text (str): The input text to preprocess.
    
    Returns:
        str: The preprocessed text.
    """
    if not isinstance(text, str):  # Check if the text is a valid string
        return ''  # Return an empty string for non-string values (like NaN)

    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation, but keep numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    
    # Rejoin the tokens into a single string
    text = ' '.join(tokens)
    
    return text


def preprocess_data(file_path):
    """
    Load data and preprocess patient text and abstracts.

    Args:
        file_path (str): The file path to the raw data.

    Returns:
        pandas.DataFrame: DataFrame with processed patient text and abstracts.
    """
    # Load the data
    df = load_data(file_path)
    
    # Apply preprocessing to patient text and abstract
    df['Processed_Patient_Text'] = df['patient']
    df['Processed_Abstract'] = df['Abstract']
    df.dropna(inplace=True)

    # Optionally, save the processed DataFrame to a new CSV
    df.to_csv('data/processed/PMC-Patients-Processed.csv', index=False)
    
    return df


if __name__ == "__main__":
    file_path = 'data/processed/PMC-Patients-with-Abstracts.csv'

    df_processed = preprocess_data(file_path)
    print("Data preprocessing complete. Processed data saved.")
