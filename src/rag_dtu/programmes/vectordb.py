import weaviate
import os
import json
import re
import pickle
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from rag_dtu.programmes.text_normalization import normalize_query
from rich.console import Console
from rich.markdown import Markdown
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import TargetVectors, MetadataQuery, Filter
import uuid

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# weaviate_api_key = "A6IFGm879tMitR94FWoffvNsBDeMTu8eZv8n"
weaviate_api_key = "YVldI0WBz6MUZVoPZA5wp5t7zalMI12jdkfm"
weaviate_url = "https://yz34awbrqlko1tvblm77g.c0.europe-west3.gcp.weaviate.cloud"

# Create client with the required grpc_port parameter
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(weaviate_api_key),  # Replace with your Weaviate Cloud key
    headers={'X-OpenAI-Api-key': os.getenv("OPENAI_API_KEY")}  # Replace with your OpenAI API key
)
print(weaviate_client.is_ready())

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
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"❌ Error: invalid input for embedding: {repr(text)}")
    
    print(f"📤 Requesting embedding for: {repr(text)}")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Initialize a specialized model for short texts (chunk names)
name_embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_name_embedding(text):
    """
    Returns a 384-dimensional embedding for short text like chunk names.
    """
    embedding = name_embedding_model.encode(text)
    return embedding.tolist()

# --- Weaviate Schema and Database Setup ---
def create_weaviate_schema():
    """
    Create Weaviate schema for chunks if it doesn't exist
    """
    # Check if collection exists
    collections = weaviate_client.collections.list_all()
    print("Existing collections:", collections)

    for collection in collections:
        if collection == "Chunk":
            print("Chunk collection already exists")
            return True
    
    # Define the collection with the new chunk structure
    chunk_collection = weaviate_client.collections.create(
        name="Chunk",
        description="A chunk of course content",
        vectorizer_config=[
            Configure.NamedVectors.none(name="name_embedding"),
            Configure.NamedVectors.none(name="content_embedding")
        ],  # We'll provide our own vectors
        properties=[
            Property(
                name="course_name",
                data_type=DataType.TEXT,
                description="The full name of the course"
            ),
            Property(
                name="chunk_name",
                data_type=DataType.TEXT,
                description="The name of the chunk"
            ),
            Property(
                name="chunk_content",
                data_type=DataType.TEXT,
                description="The content of the chunk"
            ),
            Property(
                name="chunk_url",
                data_type=DataType.TEXT,
                description="The URL of the chunk"
            )
        ]
    )
    print("Chunk collection created successfully")
    return True

def import_chunks_to_weaviate(chunk_embeddings, processed_programmes=None):
    """
    Import processed chunks into the Weaviate database using v4 client
    """
    # Make sure schema exists
    create_weaviate_schema()
    
    # Get the chunk collection
    chunk_collection = weaviate_client.collections.get("Chunk")

    # Create list to hold chunk objects
    chunk_objs = []
    count = 0
    
    for chunk_id, chunk_data in chunk_embeddings.items():
        count += 1
        print(f"Processing chunk {count}: {chunk_id}")
        
        # Get chunk metadata
        metadata = chunk_data.get('metadata', {})
        course_name = metadata.get('course_name', 'Unknown Course')
        chunk_name = metadata.get('chunk_name', 'Unknown Chunk')
        chunk_url = metadata.get('url', 'Unknown URL')
        print(f"Chunk name: {chunk_name} - Chunk URL: {chunk_url}")
        
        # Get embeddings
        name_embedding = chunk_data.get('name_embedding', [])
        content_embedding = chunk_data.get('content_embedding', [])
        
        # Generate a deterministic UUID based on chunk_id
        proper_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        
        # Get chunk content if available
        chunk_content = ""
        if processed_programmes and course_name in processed_programmes:
            chunk_content = processed_programmes[course_name].get(chunk_name, "")
        
        # Create data object
        chunk_obj = weaviate.classes.data.DataObject(
            properties={
                "course_name": course_name,
                "chunk_name": chunk_name,
                "chunk_content": chunk_content,
                "chunk_url": chunk_url
            },
            uuid=proper_uuid,
            vector={
                "name_embedding": name_embedding,
                "content_embedding": content_embedding
            }
        )
        
        print(f"Adding chunk {chunk_id} to Weaviate")
        chunk_objs.append(chunk_obj)
    
    # Insert all chunks into Weaviate
    for obj in chunk_objs:
        try:
            chunk_collection.data.insert(
                properties=obj.properties,
                uuid=obj.uuid,
                vector=obj.vector
            )
            print(f"✅ Inserted chunk for course {obj.properties.get('course_name')} - {obj.properties.get('chunk_name')}")
        except Exception as e:
            print(f"❌ Error inserting chunk: {e}")
    
    print(f"Imported {len(chunk_objs)} chunks to Weaviate")

def check_weaviate_chunks():
    """
    Check if chunks are already in Weaviate using v4 client
    """
    try:
        collections = weaviate_client.collections.list_all()
        chunks_exists = False
        
        for collection in collections:
            if collection == "Chunk":
                print("Chunk collection already exists")
                chunks_exists = True
                break
        
        if not chunks_exists:
            print("Chunk collection does not exist")
            return False
            
        # Check if collection has data
        chunk_collection = weaviate_client.collections.get("Chunk")
        response = chunk_collection.query.fetch_objects(
            limit=1
        )
        
        if len(response.objects) > 0:
            print("Collection has data")
            return True
        else:
            print("Collection exists but has no data")
            return False
    except Exception as e:
        print(f"Error checking Weaviate chunks: {e}")
        return False

# --- Weaviate Query Functions ---
def search_chunks(query, top_k=5, filter_course=None):
    """
    Search for chunks using Weaviate with hybrid search across both vector fields
    With optional filtering by course name
    """
    chunk_collection = weaviate_client.collections.get("Chunk")
    
    # Get embeddings for the query
    content_embedding = get_embedding(query)
    name_embedding = get_name_embedding(query)
    
    # Create vector query for both vector fields
    vector_query = {
        "name_embedding": name_embedding,
        "content_embedding": content_embedding
    }
    
    # Define target vectors with weights (giving more weight to content)
    weights = {
        "name_embedding": 30,  # 30% weight to name matches
        "content_embedding": 70  # 70% weight to content matches
    }
    
    # Set up filter for applied_chemistry if requested
    filter_obj = None
    if filter_course:
        print(f"Filtering by course: {filter_course}")
        filter_obj = Filter.by_property("course_name").equal(filter_course)
    
    try:
        # Perform hybrid search with optional filter
        response = chunk_collection.query.hybrid(
            query=query,
            alpha=0.75,  # Balance between vector and keyword search
            vector=vector_query,
            limit=top_k,
            target_vector=TargetVectors.manual_weights(weights),
            return_metadata=MetadataQuery(distance=True),
            return_properties=["course_name", "chunk_name", "chunk_content", "chunk_url"],
            filters=filter_obj
        )
        
        results = []
        
        if len(response.objects) > 0:
            for obj in response.objects:
                # Get distance/score if available
                distance = None
                if hasattr(obj, 'metadata') and hasattr(obj.metadata, 'distance'):
                    distance = obj.metadata.distance
                
                results.append({
                    "course_name": obj.properties.get("course_name", "Unknown Course"),
                    "chunk_name": obj.properties.get("chunk_name", "Unknown Chunk"),
                    "chunk_content": obj.properties.get("chunk_content", ""),
                    "chunk_url": obj.properties.get("chunk_url", ""),
                    "distance": distance
                })
                
            return results
        
    except Exception as e:
        print(f"Error during hybrid search: {e}")
        # Fallback with simpler search
    return []


def query_with_gpt4(user_query, programmes_info):
    """
    Constructs a prompt using the courses' preprocessed information,
    conversation history, and the user query, then uses GPT-4 to generate an answer.
    Returns both the answer and the prompt used.
    """
    # Build conversation history context
    history_context = build_history_context()
    
    # Build course info context string
    context_lines = []
    for information in programmes_info:
        course_name = information.get("course_name", "N/A")
        chunk_name = information.get("chunk_name", "Unnamed Course")
        chunk_content = information.get("chunk_content", "")
        chunk_url = information.get("chunk_url", "N/A")

        context_lines.append(
            f"Master's programme name: {course_name}\n"
            f"Chunk name: {chunk_name}\n"
            f"Chunk content: {chunk_content}\n"
            f"Chunk URL: {chunk_url}\n"
        )
    context_str = "\n\n---\n\n".join(context_lines)
    # Build the prompt using detailed sections
    prompt = f"""
You are a highly knowledgeable and polite academic assistant with expertise in all DTU Master's programmes. Your mission is to provide **accurate**, **detailed**, and **context-aware** answers about DTU Master's programmes. Speak in **first person**, as if you are personally assisting the user. Never show internal reasoning or thoughts in the reply — simply respond naturally as a helpful assistant.

You must always include the **URL of the DTU master's programme page** that was used to generate the answer. The link must be formatted as a markdown link like `[Programme Page](https://...)`, and placed **at the end** of your reply.  
Do **not include just the link** — you must also write the following phrase before it:  
> _"The information has been retrieved from the official programme page here: [Programme Page](...)"_

---

### 1. **User Query:**
- This is the most recent request from the user.
- If the query does **not include a master's programme name** or something related to the course, it could be a signal that the answer is in the conversation history.

User Query:
{user_query}

---

### 2. **Master Information:**
- This section contains official and detailed data retrieved about DTU Master's programmes.
- First, try to match the user query with information in this section, but if the user does not provide any programme name in the query, try first to figure out if it is referring to the past query, that is in the conversation history.
- It is important that there is a lot of text, so please read all parts of the text carefully to capture relevant information.

Course Information:
{context_str}

---

### 3. **Conversation History:**
- This includes recent exchanges between the user and the assistant.
- You should understand from the previous query in the following paragraph if the present query {user_query} is referring to the past query information.
- If master's programme information is unclear or missing in the query, use this section to infer context from previous user questions or previously retrieved master's programme data.

Conversation History:
{history_context}

---

### ✅ **Your Task:**
1. **Interpret the User Query:**
   - If the programme name is clearly mentioned, you can probably find the relevant details from the Master Information section and answer accordingly.
   - If not, do not be creative — first check the conversation history to figure out if the present query {user_query} is connected to the last query and if so, find the information in the conversation history. Otherwise, kindly ask the user for more details.

2. **Fallback on History:**
   - If master's programme information is not found in Master Information, check the Conversation History for a reference to the programme or topic, especially in the previous query.

3. **Ask for Clarification (if needed):**
   - If the information cannot be determined from either Master Information or Conversation History, reply in a friendly and polite tone:
     > _"I'm currently unable to identify the Master's programme. Could you kindly include the programme name so I can help more effectively?"_

4. **Include the Source URL:**
   - Always include the DTU programme page URL in the markdown response.
   - Format it as a markdown link: `[Programme Page](https://...)`
   - Do **not** include just the link. You must introduce it with the exact phrase:  
     _"The information has been retrieved from the official programme page here: [Programme Page](...)"_
   - Always place this phrase + link at the **end** of your response.

---

### ✅ **Response Style:**
- Speak **in first person**, as if personally giving suggestions or guidance.
- Always respond in **markdown** format.
- Be **factual**, **concise**, and **professional**.
- Avoid showing internal logic or thought process — just speak naturally and helpfully.
- Always include the phrase:  
  _"The information has been retrieved from the official programme page here: [Programme Page](...)"_  
  at the end of your response, as a markdown link.

---
"""

    
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are an expert academic advisor specialized in helping Master's students of Denmark University to find information regarding their courses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    answer = response.choices[0].message.content
    return answer, prompt

def get_available_programmes():
    """
    Get the list of available master's programmes from Weaviate.
    Returns a list of unique programme names.
    """
    try:
        chunk_collection = weaviate_client.collections.get("Chunk")
        response = chunk_collection.query.fetch_objects(
            limit=10000,  # Use a large limit to get all programmes
            return_properties=["course_name"]
        )
        
        programmes = set()
        for obj in response.objects:
            if "course_name" in obj.properties:
                programmes.add(obj.properties["course_name"])
        
        return sorted(list(programmes))
    except Exception as e:
        print(f"Error fetching available programmes: {e}")
        return ["Business_Analytics"]  # Fallback to default programme


def select_programme(available_programmes):
    """
    Present user with a menu to select a master's programme.
    Returns the selected programme name.
    """
    print("\nAvailable Master's Programmes:")
    print("0. All Programmes")
    
    for i, programme in enumerate(available_programmes, 1):
        print(f"{i}. {programme.replace('_', ' ')}")
    
    while True:
        try:
            choice = input("\nSelect a programme number (0 for All): ")
            
            # Check if user wants to exit programme selection
            if choice.lower() in ["exit", "quit", "cancel"]:
                return "All Programmes"
            
            choice_num = int(choice)
            
            if choice_num == 0:
                print("\nSelected: All Programmes")
                return "All Programmes"
            elif 1 <= choice_num <= len(available_programmes):
                selected = available_programmes[choice_num - 1]
                print(f"\nSelected: {selected.replace('_', ' ')}")
                return selected
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
            


def interactive_chat(processed_embedding_path, processed_courses_path):
    """
    Run an interactive chat session that allows testing the chunk retrieval system.
    Users can select a specific master's programme to filter their queries.
    """
    print("Welcome to the DTU Course Assistant!")
    print("Ask me anything about DTU master's programmes. Type 'exit' to quit.\n")
    
    # Check if chunks exist in Weaviate
    with open(processed_courses_path, "r", encoding="utf-8") as f:
        processed_courses = json.load(f)
    
    with open(processed_embedding_path, "r", encoding="utf-8") as f:
        processed_embeddings = json.load(f)
    
    # Check if courses are in Weaviate, import if not
    if not check_weaviate_chunks():
        print("Importing courses to Weaviate...")
        import_chunks_to_weaviate(processed_embeddings, processed_courses)
        print("Courses imported successfully.")
    
    # Get available master's programmes
    available_programmes = get_available_programmes()
    
    # Let the user select a master's programme
    selected_programme = select_programme(available_programmes)
    
    # Main chat loop
    while True:
        # Get user query
        user_query = input("\nYour question: ")
        
        # Exit condition
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Thanks for chatting! Goodbye!")
            break
        
        # Change programme selection
        if user_query.lower() in ["change programme", "switch programme", "select programme"]:
            selected_programme = select_programme(available_programmes)
            continue
        
        # Normalize the query
        normalized_query, _ = normalize_query(user_query)

        # Use the selected programme as filter
        filter_course = selected_programme if selected_programme != "All Programmes" else None
        
        # Search for relevant chunks with filter
        print("Searching for relevant information...")
        retrieved_results = search_chunks(normalized_query, top_k=5, filter_course=filter_course)
        
        if not retrieved_results:
            print("Sorry, I couldn't find any relevant information about what you asked.")
            continue
        
        # Format retrieved information
        retrieved_info_str = ""
        
        print("\nRelevant information found:")
        for i, result in enumerate(retrieved_results, 1):
            print(f"Result {i}:")
            print(f"Course Name: {result['course_name']}")
            print(f"Chunk Name: {result['chunk_name']}")
            course_name = result["course_name"]
            chunk_name = result["chunk_name"]
            chunk_content = result["chunk_content"]
            chunk_url = result["chunk_url"]
            
            # Add to the retrieved info string for conversation history
            retrieved_info_str += f"Programme: {course_name}\n"
            retrieved_info_str += f"Section: {chunk_name}\n\n"
            retrieved_info_str += f"Chunk content: {chunk_content}\n\n"
            retrieved_info_str += f"Chunk URL: {chunk_url}\n\n"
            retrieved_info_str += "---\n\n"
        
        answer, _ = query_with_gpt4(user_query, retrieved_results)
        print("\n 🧠 Assistant 🧠:")
        display_markdown(answer)
        print("\n---\n")
        # Update conversation history for context
        update_conversation_history(user_query, retrieved_info_str)



        
if __name__ == "__main__":
    # Load chunked embeddings from JSON file
    
    # Uncomment to import chunks to Weaviate
    #import_chunks_to_weaviate(chunked_embeddings, processed_programmes)
    
    interactive_chat("data/chunked_programme_embeddings.json", "data/processed_programmes.json")
    weaviate_client.close()