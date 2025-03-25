import faiss
import numpy as np
import json
import os
import pickle
import re
import openai
from sentence_transformers import SentenceTransformer

# Set your OpenAI API key
openai.api_key = "your key here"

# --- Helper Functions and Models ---

def get_embedding(text, model="text-embedding-3-small"):
    """
    Returns the embedding vector for the given text using the specified OpenAI model.
    """
    response = openai.Embedding.create(input=text, model=model)
    return response["data"][0]["embedding"]

# Initialize a specialized model for short texts (course names)
name_embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_name_embedding(text):
    """
    Returns a 384-dimensional embedding for short text like course names.
    """
    embedding = name_embedding_model.encode(text)
    return embedding.tolist()

def extract_filters_from_query(query):
    """
    Extracts a 5-digit course code from the query, if present.
    """
    code_match = re.search(r"\b\d{5}\b", query)
    code = code_match.group(0) if code_match else None
    filters = {}
    if code:
        filters['course_code'] = code
    return filters

# --- Hybrid FAISS Vector Database Class ---
class HybridFAISSVectorDB:
    def __init__(self, content_dim, name_dim,
                 content_index_path="faiss_content.index",
                 name_index_path="faiss_name.index",
                 metadata_path="metadata.pkl"):
        self.content_dim = content_dim
        self.name_dim = name_dim
        self.content_index_path = content_index_path
        self.name_index_path = name_index_path
        self.metadata_path = metadata_path
        
        self.content_index = None
        self.name_index = None
        
        # Dictionaries for mapping course IDs to index positions
        self.content_id_to_index = {}
        self.content_index_to_id = {}
        self.name_id_to_index = {}
        self.name_index_to_id = {}
        
        self.metadata = {}
        
        if os.path.exists(content_index_path) and os.path.exists(name_index_path) and os.path.exists(metadata_path):
            self._load()
        else:
            # If the indexes do not exist, raise an error.
            raise FileNotFoundError("The FAISS indexes or metadata file was not found. Make sure you have created the vector DB first.")
    
    def search(self, query, top_k=10, filters=None,
               weight_content=0.7, weight_name=0.3):
        """
        Searches for courses based on the query.
        - If filters include a course code, perform an exact lookup.
        - Otherwise, perform a hybrid semantic search using both embeddings.
        
        The weights (weight_content and weight_name) define the contribution of each embedding's distance.
        """
        # First: Exact lookup based on course code
        if filters and 'course_code' in filters:
            filtered_ids = []
            for course_id, meta in self.metadata.items():
                if str(meta.get('course_code', '')).zfill(5) == filters['course_code']:
                    filtered_ids.append(course_id)
            if filtered_ids:
                results = []
                for course_id in filtered_ids:
                    results.append({
                        "id": course_id,
                        "distance": 0.0,  # Exact match
                        "metadata": self.metadata[course_id]
                    })
                return results[:top_k]
        
        # Otherwise: Hybrid semantic search
        query_content_embedding = np.array(get_embedding(query)).astype('float32').reshape(1, -1)
        query_name_embedding = np.array(get_name_embedding(query)).astype('float32').reshape(1, -1)
        
        distances_content, indices_content = self.content_index.search(query_content_embedding, top_k)
        distances_name, indices_name = self.name_index.search(query_name_embedding, top_k)
        
        combined_scores = {}
        # Process content index results
        for i in range(len(indices_content[0])):
            idx = indices_content[0][i]
            if idx < 0:
                continue
            course_id = self.content_index_to_id.get(idx)
            if course_id is None:
                continue
            combined_scores[course_id] = combined_scores.get(course_id, 0) + weight_content * distances_content[0][i]
        
        # Process name index results
        for i in range(len(indices_name[0])):
            idx = indices_name[0][i]
            if idx < 0:
                continue
            course_id = self.name_index_to_id.get(idx)
            if course_id is None:
                continue
            combined_scores[course_id] = combined_scores.get(course_id, 0) + weight_name * distances_name[0][i]
        
        # Sort the courses by combined score (lower is better)
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1])
        
        final_results = []
        for course_id, score in sorted_results[:top_k]:
            final_results.append({
                "id": course_id,
                "distance": score,
                "metadata": self.metadata.get(course_id, {})
            })
        
        return final_results

    def _load(self):
        self.content_index = faiss.read_index(self.content_index_path)
        self.name_index = faiss.read_index(self.name_index_path)
        with open(self.metadata_path, "rb") as f:
            data = pickle.load(f)
            self.metadata = data.get('metadata', {})
            self.content_id_to_index = data.get('content_id_to_index', {})
            self.content_index_to_id = data.get('content_index_to_id', {})
            self.name_id_to_index = data.get('name_id_to_index', {})
            self.name_index_to_id = data.get('name_index_to_id', {})
        print("🔁 FAISS indexes and metadata loaded.")

# --- GPT-4 Query Function ---

def query_with_gpt4(user_query, courses_info):
    """
    Constructs a prompt using the courses' preprocessed information and the user query,
    then uses GPT-4 to generate a comprehensive answer.
    
    courses_info should be a dictionary mapping course IDs to course details.
    """
    # Build a context string with course code, title, and preprocessed text.
    context_lines = []
    for course_id, info in courses_info.items():
        metadata = info.get("metadata", {})
        course_code = metadata.get("course_code", "N/A")
        course_name = metadata.get("course_name", "Unnamed Course")
        preprocessed_text = info.get("preprocessed_course", "")
        context_lines.append(
            f"Course Code: {course_code}\nTitle: {course_name}\nDetails: {preprocessed_text}\n"
        )
    context_str = "\n---\n".join(context_lines)
    
    prompt = f"""You are an expert academic advisor.
Using the following course information, please answer the user's query in detail.

Course Information:
{context_str}

User Query:
{user_query}

Provide a comprehensive answer that is both factual and helpful."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are an expert academic advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    return answer

# --- Main Query Flow ---
if __name__ == "__main__":
    # Load the processed courses JSON (which contains the preprocessed_course text)
    with open("processed_courses.json", "r", encoding="utf-8") as f:
        processed_courses = json.load(f)
    
    # Initialize the vector database from saved files.
    vector_db = HybridFAISSVectorDB(content_dim=1536, name_dim=384)
    
    # Example user query
    user_query = "What is the suggested literature for model-based machine learning course?"
    filters = extract_filters_from_query(user_query)
    
    # Search the vector database using the user query
    search_results = vector_db.search(user_query, top_k=5, filters=filters)
    
    print(f"\n🔎 Query: {user_query}")
    if filters:
        print(f"🔍 Detected filters: {filters}")
    
    # Gather detailed course info from processed_courses using the IDs from search results.
    retrieved_courses = {}
    for result in search_results:
        course_id = result["id"]
        if course_id in processed_courses:
            retrieved_courses[course_id] = processed_courses[course_id]
        else:
            print(f"⚠️ Warning: Course ID {course_id} not found in processed_courses.json.")
    
    # Pass the retrieved course information along with the query to GPT-4 for a detailed answer.
    answer = query_with_gpt4(user_query, retrieved_courses)
    
    print("\n📝 GPT-4 Answer:")
    print(answer)
