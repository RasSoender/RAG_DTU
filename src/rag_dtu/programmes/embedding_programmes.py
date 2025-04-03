from openai import OpenAI
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    """
    Returns the embedding vector for the given text using the specified OpenAI model.
    """
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# Initialize a specialized model for short texts (programme names)
name_embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_name_embedding(text):
    """
    Returns an embedding for short text like programme names using a specialized model.
    This model directly outputs a 384-dimensional vector.
    """
    embedding = name_embedding_model.encode(text)
    return embedding.tolist()

def process_programme(json_file_path, use_postprocessed=True):
    """
    Processes a JSON file with programme entries, generating embeddings for each programme.
    Generates a full embedding for programme content via OpenAI and a specialized embedding for programme names.
    
    Returns a dictionary with programme IDs as keys and a dict with content embedding, name embedding, and metadata.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        programmes = json.load(f)

    embeddings = {}
    for programme_name, data in programmes.items():
        print(f"Processing programme: {programme_name}")

        name_embedding = get_name_embedding(programme_name)

        embeddings[programme_name] = {
            "name_embedding": name_embedding,
            "metadata": data.get("metadata", {})
        }

        for key, value in data.items():
            if isinstance(value, str) and key != "metadata":
                embeddings[key] = get_embedding(value)  

        print(f"Processed programme with specialized name embedding for '{programme_name}'")

    return embeddings

if __name__ == "__main__":
    # Specify the path to your JSON file containing programme entries
    json_file_path = "data/processed_programmes.json"

    # Process the programmes and generate embeddings
    programme_embeddings = process_programme(json_file_path)

    # Save the results to a JSON file for later use
    with open("data/programme_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(programme_embeddings, f, indent=2)

    print("Content and specialized name embeddings have been generated and saved.")
