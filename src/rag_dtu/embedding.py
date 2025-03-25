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

# Initialize a specialized model for short texts (course names)
name_embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_name_embedding(text):
    """
    Returns an embedding for short text like course names using a specialized model.
    This model directly outputs a 384-dimensional vector.
    """
    embedding = name_embedding_model.encode(text)
    return embedding.tolist()

def process_courses(json_file_path, use_postprocessed=True):
    """
    Processes a JSON file with course entries, generating embeddings for each course.
    Generates a full embedding for course content via OpenAI and a specialized embedding for course names.
    
    Returns a dictionary with course IDs as keys and a dict with content embedding, name embedding, and metadata.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        courses = json.load(f)

    embeddings = {}
    for course_id, data in courses.items():
        # Use postprocessed text if available and requested; otherwise, use preprocessed text
        if use_postprocessed and "postprocessed_course" in data:
            text = data["postprocessed_course"]
        elif "preprocessed_course" in data:
            text = data["preprocessed_course"]
        else:
            continue  # Skip if there's no text available to embed

        # Get the course name from metadata or use a default
        course_name = data.get("metadata", {}).get("title", f"Course {course_id}")

        # For course titles, include department and course code in the embedding if available
        department = data.get("metadata", {}).get("department", "")
        course_code = data.get("metadata", {}).get("course_code", "")

        # Create an enhanced name string with extra context for better embedding
        enhanced_name = f"{course_name}"
        if department:
            enhanced_name += f" - {department}"
        if course_code:
            enhanced_name += f" (Code: {course_code})"

        # Generate embeddings: full for course content and specialized for course names
        content_embedding = get_embedding(text)
        name_embedding = get_name_embedding(enhanced_name)

        embeddings[course_id] = {
            "content_embedding": content_embedding,
            "name_embedding": name_embedding,
            "metadata": data.get("metadata", {})
        }
        print(f"Processed course {course_id} with specialized name embedding for '{course_name}'")

    return embeddings

if __name__ == "__main__":
    # Specify the path to your JSON file containing course entries
    json_file_path = "data/processed_courses.json"

    # Process the courses and generate embeddings
    course_embeddings = process_courses(json_file_path, use_postprocessed=True)

    # Save the results to a JSON file for later use
    with open("course_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(course_embeddings, f)

    print("Content and specialized name embeddings have been generated and saved.")
