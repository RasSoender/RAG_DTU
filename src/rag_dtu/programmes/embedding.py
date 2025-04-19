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

# Initialize a specialized model for short texts (program names and chunk names)
name_embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_name_embedding(text):
    """
    Returns an embedding for short text using a specialized model.
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

def process_programme_chunks(json_file_path, json_summarized_path):
    """
    Processes JSON files with programme entries, merges them, and generates 
    chunked embeddings for each field.
    
    The output dictionary has the following structure:
    
    {
      "chunk_id_1": {
          "name_embedding": [ ... ],    # embedding for the chunk name
          "content_embedding": [ ... ],  # embedding for the chunk content
          "metadata": {
              "course_name": "Applied Chemistry",
              "chunk_name": "programme_specification"
          }
      },
      "chunk_id_2": { ... }
    }
    """
    # Load the raw and summarized JSON data
    with open(json_file_path, "r", encoding="utf-8") as f:
        programmes = json.load(f)

    with open(json_summarized_path, "r", encoding="utf-8") as f:
        summarized_programmes = json.load(f)

    # Merge the two JSON files
    merged_programmes = merge_files(programmes, summarized_programmes)

    # Optionally, save the merged data for inspection
    with open("data/merged_programmes.json", "w", encoding="utf-8") as f:
        json.dump(merged_programmes, f, indent=2)

    chunked_embeddings = {}
    chunk_counter = 0
    
    for programme_name, data in merged_programmes.items():
        print(f"Processing programme: {programme_name}")

        # Process each field in the merged programme data as a separate chunk
        for field_name, field_content in data.items():
            # Skip if the field is not a non-empty string
            if not isinstance(field_content, str) or field_content.strip() == "":
                continue
            if field_name == "programme_name":
                continue
            # Create a unique identifier for this chunk
            chunk_id = f"chunk_{chunk_counter}"
            chunk_counter += 1
            
            # Create entry for this chunk
            chunk_entry = {
                "name_embedding": get_name_embedding(field_name),
                "content_embedding": get_embedding(field_content),
                "metadata": {
                    "course_name": programme_name,
                    "chunk_name": field_name
                },
                # Optionally include the raw content for verification
                # "raw_content": field_content
            }
            
            chunked_embeddings[chunk_id] = chunk_entry
            print(f"Processed chunk: {programme_name} - {field_name}")

    return chunked_embeddings

if __name__ == "__main__":
    # Specify the paths to your JSON files
    json_file_path = "data/processed_programmes.json"
    summarized_path = "data/summarized_programmes.json"

    # Process the programmes and generate chunked embeddings
    chunked_embeddings = process_programme_chunks(json_file_path, summarized_path)

    # Save the results to a JSON file for later use
    with open("data/chunked_programme_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(chunked_embeddings, f, indent=2)

    print(f"Chunked embeddings have been generated and saved. Total chunks: {len(chunked_embeddings)}")