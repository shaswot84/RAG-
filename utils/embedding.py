import requests
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

def generate_embeddings(text, model = "text-embedding-3-small"):
    url = "https://api.euron.one/api/v1/euri/alpha/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "input": text,
        "model": model
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    # Convert to numpy array for vector operations
    embedding = np.array(data['data'][0]['embedding'])
    return embedding