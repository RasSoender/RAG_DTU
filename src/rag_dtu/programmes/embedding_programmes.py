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

def merge_files(normal, summarized):
    """
    Merges two JSON files, normal and summarized, into a single dictionary.
    The keys are the programme names, and the values are dictionaries containing
    the content and metadata.
    """
    merged = {}
    for programme_name, data in summarized.items():
        merged[programme_name] = {}
        for key, value in data.items():
            if not value or value.strip() == "":
                merged[programme_name][key] = normal.get(programme_name, {}).get(key, "")
            else:
                merged[programme_name][key] = value
    return merged


def process_programme(json_file_path, json_summarized_path, use_postprocessed=True):
    """
    Processes a JSON file with programme entries, generating embeddings for each programme.
    Embeddings are stored directly under the programme, with a 'metadata' dict containing
    field-level metadata for each field.
    
    Returns a dictionary with programme names as keys and all embeddings + metadata.
    """
    import json

    with open(json_file_path, "r", encoding="utf-8") as f:
        programmes = json.load(f)

    with open(json_summarized_path, "r", encoding="utf-8") as f:
        summarized_programmes = json.load(f)

    # Merge the two JSON files
    merged_programmes = merge_files(programmes, summarized_programmes)

    # Save the merged data
    with open("data/merged_programmes.json", "w", encoding="utf-8") as f:
        json.dump(merged_programmes, f, indent=2)

    embeddings = {}
    for programme_name, data in merged_programmes.items():
        print(f"Processing programme: {programme_name}")

        # Initialize container for this programme
        programme_entry = {
            "name_embedding": get_name_embedding(programme_name),
            "metadata": {}
        }

        for key, value in data.items():
            if isinstance(value, str) and value.strip() != "" and key != "metadata":
                field_embedding = get_embedding(value)
                programme_entry[key] = field_embedding
                programme_entry["metadata"][key] = {
                    "programme_name": programme_name
                }

        embeddings[programme_name] = programme_entry
        print(f"Processed: {programme_name}")

    return embeddings



if __name__ == "__main__":
    # Specify the path to your JSON file containing programme entries
    json_file_path = "data/processed_programmes.json"

    summarized_path = "data/summarized_programmes.json"

    # Process the programmes and generate embeddings
    programme_embeddings = process_programme(json_file_path, summarized_path)

    # Save the results to a JSON file for later use
    with open("data/programme_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(programme_embeddings, f, indent=2)

    print("Content and specialized name embeddings have been generated and saved.")
