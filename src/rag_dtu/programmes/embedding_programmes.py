from openai import OpenAI
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_dtu.config import get_openai_api_key

# Set your OpenAI API key
client = OpenAI(api_key=get_openai_api_key())

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


import json
import re
import string

def process_programme(json_file_path, json_summarized_path, use_postprocessed=True):
    """
    Processes JSON files with programme entries, merges them, and generates embeddings for each field.
    For each programme, the output dictionary has the following structure:
    
    {
      "Applied Chemistry": {
          "programme_specification": [ ... ],   # embedding for that field
          "duration": [ ... ],
          "admission_requirements": [ ... ],
          ... (other fields) ...,
          "name_embedding": [ ... ],            # embedding for the programme name
          "metadata": {"course_name": "Applied Chemistry"}
      },
      "Other Subject": { ... }
    }
    
    Each non-empty string field in the merged programme data is embedded using get_embedding().
    The programme name itself is embedded using get_name_embedding().
    """
    # Load the raw and summarized JSON data
    with open(json_file_path, "r", encoding="utf-8") as f:
        programmes = json.load(f)

    with open(json_summarized_path, "r", encoding="utf-8") as f:
        summarized_programmes = json.load(f)

    # Merge the two JSON files (assuming merge_files is defined)
    merged_programmes = merge_files(programmes, summarized_programmes)

    # Optionally, save the merged data for inspection
    with open("data/merged_programmes.json", "w", encoding="utf-8") as f:
        json.dump(merged_programmes, f, indent=2)

    embeddings = {}
    for programme_name, data in merged_programmes.items():
        print(f"Processing programme: {programme_name}")

        # Initialize the nested dictionary for this programme.
        programme_entry = {}

        # Process each field in the merged programme data
        for key, value in data.items():
            # Skip if the value is not a non-empty string.
            if key == "programme_name": 
                # Skip the programme name field itself, as we'll handle it separately.
                continue
            if isinstance(value, str) and value.strip() != "":
                # Optionally, you could perform preprocessing on the text here (e.g., lowercasing, punctuation removal)
                # For example:
                
                # Generate an embedding for this field.
                programme_entry[key] = get_embedding(value)
        
        # Add the specialised name embedding for the programme name.
        programme_entry["name_embedding"] = get_name_embedding(data["programme_name"])
        
        # Include a metadata dict with only the course name.
        programme_entry["metadata"] = {"course_name": programme_name}

        embeddings[programme_name] = programme_entry
        print(f"Processed: {programme_name}")

    return embeddings

# Example usage:
# embeddings = process_programme("programmes.json", "summarized_programmes.json")
# The resulting embeddings dict will have a nested structure where each field's embedding is stored.



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
