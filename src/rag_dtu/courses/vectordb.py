import faiss
import numpy as np
import json
import os
import pickle
import re
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from text_normalization import normalize_query
from rich.console import Console
from rich.markdown import Markdown

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Short-Term Memory (Conversation History) ---
conversation_history = []

def update_conversation_history(user_query, retrieved_info, max_history=2):
    """
    Update the conversation history with the latest exchange.
    Only the last `max_history` exchanges are kept.
    Instead of storing the assistant's previous prompt, we store the retrieved course information.
    """
    conversation_history.append({
        "user": user_query,
        "retrieved": retrieved_info
    })
    if len(conversation_history) > max_history:
        conversation_history[:] = conversation_history[-max_history:]

def build_history_context():
    """
    Build a string representing the conversation history.
    Here we include the user query and the previously retrieved course information.
    """
    history_lines = []
    for exchange in conversation_history:
        history_lines.append(f"User: {exchange['user']}\nRetrieved Info: {exchange['retrieved']}")
    return "\n\n".join(history_lines)

def display_markdown(answer):
    """
    Display a markdown-formatted string in a styled way using the rich library.
    """
    console = Console()
    md = Markdown(answer)
    console.print(md)


# --- Helper Functions and Models ---
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
    Returns a 384-dimensional embedding for short text like course names.
    """
    embedding = name_embedding_model.encode(text)
    return embedding.tolist()

def create_vector_database(processed_courses, 
                            content_index_path, 
                            name_index_path, 
                            metadata_path):
    """
    Create FAISS vector indexes for course content and names.
    """
    content_embeddings = []
    name_embeddings = []
    metadata = {}
    content_id_to_index = {}
    content_index_to_id = {}
    name_id_to_index = {}
    name_index_to_id = {}

    for course_id, course_data in processed_courses.items():
        print(f'Processing course {course_id}')
        
        # Get course metadata
        course_meta = course_data.get('metadata', {})
        
        # Get content embedding
        content_text = course_data.get('preprocessed_course', '')
        content_emb = get_embedding(content_text)
        content_embeddings.append(content_emb)
        
        # Get name embedding
        course_name = course_meta.get('course_name', '')
        name_emb = get_name_embedding(course_name)
        name_embeddings.append(name_emb)

        # Store metadata and index mappings
        metadata[course_id] = course_meta
        content_id_to_index[course_id] = len(content_embeddings) - 1
        content_index_to_id[len(content_embeddings) - 1] = course_id
        name_id_to_index[course_id] = len(name_embeddings) - 1
        name_index_to_id[len(name_embeddings) - 1] = course_id

    # Convert to numpy arrays
    content_embeddings = np.array(content_embeddings).astype('float32')
    name_embeddings = np.array(name_embeddings).astype('float32')

    # Create FAISS indexes
    content_index = faiss.IndexFlatL2(content_embeddings.shape[1])
    name_index = faiss.IndexFlatL2(name_embeddings.shape[1])

    content_index.add(content_embeddings)
    name_index.add(name_embeddings)

    # Save FAISS indexes
    faiss.write_index(content_index, content_index_path)
    faiss.write_index(name_index, name_index_path)

    # Save metadata and mappings
    with open(metadata_path, "wb") as f:
        pickle.dump({
            'metadata': metadata,
            'content_id_to_index': content_id_to_index,
            'content_index_to_id': content_index_to_id,
            'name_id_to_index': name_id_to_index,
            'name_index_to_id': name_index_to_id
        }, f)

    print("Vector database created successfully!")
    print(f"Content embeddings: {content_embeddings.shape}")
    print(f"Name embeddings: {name_embeddings.shape}")

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
                 content_index_path,
                 name_index_path,
                 metadata_path):
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
            raise FileNotFoundError("The FAISS indexes or metadata file was not found. Make sure you have created the vector DB first.")

    def search(self, query, top_k=3, filters=None,
               weight_content=0.7, weight_name=0.3):
        """
        Searches for courses based on the query.
        - If filters include a course code, perform an exact lookup.
        - Otherwise, perform a hybrid semantic search using both embeddings.
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
        print("FAISS indexes and metadata loaded.")

# --- GPT-4 Query Function ---
def query_with_gpt4(user_query, courses_info):
    """
    Constructs a prompt using the courses' preprocessed information,
    conversation history, and the user query, then uses GPT-4 to generate an answer.
    Returns both the answer and the prompt used.
    """
    # Build conversation history context
    history_context = build_history_context()
    
    # Build course info context string
    context_lines = []
    for course_id, info in courses_info.items():
        metadata = info.get("metadata", {})
        course_code = metadata.get("course_code", "N/A")
        course_name = metadata.get("course_name", "Unnamed Course")
        preprocessed_text = info.get("preprocessed_course", "")
        context_lines.append(
            f"Course Code: {course_code}\nTitle: {course_name}\nDetails: {preprocessed_text}"
        )
    context_str = "\n\n---\n\n".join(context_lines)
    
    # Build the prompt using detailed sections
    prompt = f"""
You are a highly knowledgeable and polite academic assistant with expertise in all DTU courses. Your mission is to provide **accurate**, **detailed**, and **context-aware** answers about DTU courses. Speak in **first person**, as if you are personally assisting the user. Never show internal reasoning or thoughts in the reply — simply respond naturally as a helpful assistant.

---

### 1. **User Query:**
- This is the most recent request from the user.
- If the query does **not include a course name or course code** or something related to the course, it could be a signal that the answer is in the conversation history.

User Query:
{user_query}

---

### 2. **Course Information:**
- This section contains official and detailed data about DTU courses.
- First, try to match the user query with information in this section, but if the user do not provide any course name or code in the query, try firstly to figure out if it is referring to the past query, that is in the conversation history.

Course Information:
{context_str}

---

### 3. **Conversation History:**
- This includes recent exchanges between the user and the assistant.
-You should understand from the previous query in the following paragraph if the present query {user_query} is referring to the past query informations.
- If course information is unclear or missing in the query, use this section to infer context from previous user questions or previously retrieved course data.

Conversation History:
{history_context}

---

### ✅ **Your Task:**
1. **Interpret the User Query:**
   - If the course name or code is clearly mentioned, probably you can find the relevant course details from the Course Information section and answer accordingly.
   - If not, do not be creative, firstly check the conversation history for figuring out if the present query {user_query} is connected to the last query and so you can find information in the conversation history, otherwise simply ask more information.

2. **Fallback on History:**
   - If a course information is not found in Course Information, check the Conversation History for a reference to the course or topic, mostly in the previous query.

3. **Ask for Clarification (if needed):**
   - If the information cannot be determined from either Course Information or Conversation History, reply in a friendly and polite tone:
     > _"I'm currently unable to identify the course. Could you kindly include the course code so I can help more effectively?"_

---

### ✅ **Response Style:**
- Speak **in first person**, as if personally giving suggestions or guidance.
- Always respond in **markdown** format.
- Be **factual**, **concise**, and **professional**.
- Avoid showing internal logic or thought process — just speak naturally and helpfully.

---
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are an expert academic advisor specialized in helping Master's students of Denmark University to find information regarding their courses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    answer = response.choices[0].message.content
    return answer, prompt


# --- Interactive Chat Function ---
def interactive_chat(vector_db, processed_courses):
    print("Starting interactive chat session. Type 'exit' or 'quit' to end.")
    while True:
        user_query = input("\n 👨‍💼 User 👨‍💼: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            print("Exiting interactive chat.")
            print("Goodbye!")
            break

        normalized_query, _ = normalize_query(user_query)
        filters = extract_filters_from_query(normalized_query)

        # Search the vector database using the user query
        search_results = vector_db.search(normalized_query, top_k=5, filters=filters)

        # Gather detailed course info from processed_courses using the IDs from search results.
        retrieved_courses = {}
        for result in search_results:
            course_id = result["id"]
            if course_id in processed_courses:
                retrieved_courses[course_id] = processed_courses[course_id]
            else:
                print(f"Warning: Course ID {course_id} not found in processed_courses.")

        # Build a summary string of retrieved course information for conversation history.
        retrieved_courses_str = ""
        for course_id, info in retrieved_courses.items():
            metadata = info.get("metadata", {})
            course_code = metadata.get("course_code", "N/A")
            course_name = metadata.get("course_name", "Unnamed Course")
            retrieved_courses_str += f"Course Code: {course_code}\nTitle: {course_name}\n\n"
        
        # Get GPT-4 answer along with the current prompt (for logging purposes)
        answer, _ = query_with_gpt4(user_query, retrieved_courses)
        print("\n 🧠 Assistant 🧠:")
        display_markdown(answer)
        print("\n---\n")

        # Update the conversation history with the user query and the retrieved course summary
        update_conversation_history(user_query, retrieved_courses)


# --- Main Flow ---
if __name__ == "__main__":
    # Load the processed courses JSON (which contains the preprocessed_course text)
    with open("data/processed_courses.json", "r", encoding="utf-8") as f:
        processed_courses = json.load(f)

    content_index_path = "data/vector_db/faiss_content.index"
    name_index_path = "data/vector_db/faiss_name.index"
    metadata_path = "data/vector_db/metadata.pkl"

    # Create the vector database if it does not exist
    if not os.path.exists(content_index_path) or not os.path.exists(name_index_path) or not os.path.exists(metadata_path):
        create_vector_database(processed_courses, content_index_path, name_index_path, metadata_path)

    # Initialize the vector database from saved files.
    vector_db = HybridFAISSVectorDB(
        content_dim=1536, 
        name_dim=384, 
        content_index_path=content_index_path, 
        name_index_path=name_index_path, 
        metadata_path=metadata_path
    )

    # Start the interactive chat session.
    interactive_chat(vector_db, processed_courses)


weaviate_url = "https://yz34awbrqlko1tvblm77g.c0.europe-west3.gcp.weaviate.cloud"
weaviate_api_key = "YVldI0WBz6MUZVoPZA5wp5t7zalMI12jdkfm"