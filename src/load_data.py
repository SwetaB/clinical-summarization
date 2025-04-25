import pandas as pd
import os
import requests
from xml.etree import ElementTree
import time

def load_data(file_path, num_rows=25000):
    """
    Load data from the provided CSV file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    
    df = pd.read_csv(file_path)
    
    return df.head(num_rows)

def fetch_abstract(pmid, delay=0.5):
    """
    Fetch the abstract for a given PMID from PubMed using E-utilities.
    
    Args:
        pmid (str): PubMed ID
        delay (int): Delay in seconds between requests to avoid hitting the rate limit.

    Returns:
        str: Abstract text, or None if not found.
    """
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': pmid,
        'retmode': 'xml',
        'rettype': 'abstract',
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parse the XML response to get the abstract
        tree = ElementTree.fromstring(response.content)
        abstract = tree.find(".//AbstractText")
        if abstract is not None:
            return abstract.text
        else:
            return "No abstract available"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching abstract for PMID {pmid}: {e}")
        return None
    finally:
        # Sleep to avoid hitting the rate limit
        time.sleep(delay) 

def add_abstracts_to_df(df):
    """
    Adds abstracts to the DataFrame by fetching them for each PMID.
    """
    abstracts = []
    for pmid in df['PMID']:
        abstract = fetch_abstract(pmid)
        abstracts.append(abstract)
    
    # Add the abstracts as a new column in the DataFrame
    df['Abstract'] = abstracts
    return df

# Load the data
file_path = 'data/raw/PMC-Patients.csv'  # Adjust path if necessary
df = load_data(file_path)

# Add abstracts to the DataFrame
df_with_abstracts = add_abstracts_to_df(df)

# Save the DataFrame with abstracts as a new CSV (optional)
df_with_abstracts.to_csv('data/processed/PMC-Patients-with-Abstracts.csv', index=False)
