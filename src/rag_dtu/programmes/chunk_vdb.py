import weaviate
import os
import json
import re
import pickle
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from text_normalization_programmes import normalize_query
from rich.console import Console
from rich.markdown import Markdown
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import TargetVectors, MetadataQuery, Filter
import uuid

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
weaviate_api_key = "4JjcaYcEYBUzb46TpPu6f1qR5CjVJXb5wFB7"
weaviate_url = "fjax7aot34bgxxo433ma.c0.europe-west3.gcp.weaviate.cloud"

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
                "chunk_content": chunk_content
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
        "name_embedding": 70,  # 30% weight to name matches
        "content_embedding": 30  # 70% weight to content matches
    }
    
    # Set up filter for applied_chemistry if requested
    filter_obj = None
    if filter_course:
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
            return_properties=["course_name", "chunk_name", "chunk_content"],
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
                    "distance": distance
                })
                
            return results
        
    except Exception as e:
        print(f"Error during hybrid search: {e}")
        # Fallback with simpler search
        try:
            print("Trying simplified search...")
            # Include filter in simplified search too
            response = chunk_collection.query.hybrid(
                query=query,
                alpha=0.75,
                vector={"content_embedding": content_embedding},  # Just use content embedding
                limit=top_k,
                return_metadata=MetadataQuery(distance=True),
                return_properties=["course_name", "chunk_name", "chunk_content"],
                filters=filter_obj
            )
            
            if len(response.objects) > 0:
                return [{
                    "course_name": obj.properties.get("course_name", "Unknown Course"),
                    "chunk_name": obj.properties.get("chunk_name", "Unknown Chunk"),
                    "chunk_content": obj.properties.get("chunk_content", ""),
                    "distance": obj.metadata.distance if hasattr(obj, 'metadata') else None
                } for obj in response.objects]
        except Exception as e2:
            print(f"Error during simplified search: {e2}")
    
    return []

def interactive_chat():
    """
    Run an interactive chat session that allows testing the chunk retrieval system.
    """
    print("Welcome to the DTU Course Assistant!")
    print("Ask me anything about DTU master's programmes. Type 'exit' to quit.\n")
    
    # Check if chunks exist in Weaviate
    if not check_weaviate_chunks():
        print("No chunks found in Weaviate. Please make sure chunks are imported first.")
        return
    else:
        print("Chunks found in Weaviate, ready to query.")
    
    # Set the filter to Applied Chemistry
    applied_chemistry_filter = "Applied Chemistry"
    print(f"Filtering results to only show information about: {applied_chemistry_filter}")
    
    while True:
        # Get user query
        user_query = input("\nYour question: ")
        
        # Exit condition
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Thanks for chatting! Goodbye!")
            break
        
        # Normalize the query
        normalized_query, _ = normalize_query(user_query)
        
        # Search for relevant chunks with filter
        print("Searching for relevant information...")
        retrieved_results = search_chunks(normalized_query, top_k=5, filter_course=applied_chemistry_filter)
        
        if not retrieved_results:
            print("Sorry, I couldn't find any relevant information about Applied Chemistry.")
            continue
        
        
        # Format retrieved information
        retrieved_info_str = ""
        
        print("\nRelevant information found:")
        for i, result in enumerate(retrieved_results, 1):
            course_name = result["course_name"]
            chunk_name = result["chunk_name"]
            chunk_content = result["chunk_content"]
            distance = result.get("distance", None)
            
            print(f"{i}. {course_name} - {chunk_name}")
            if distance is not None:
                print(f"   Relevance score: {1 - distance:.2f}")
            
            # Preview of content (first 100 characters)
            content_preview = chunk_content[:100] + "..." if len(chunk_content) > 100 else chunk_content
            print(f"   Preview: {content_preview}")
            
            # Add to the retrieved info string for conversation history
            retrieved_info_str += f"Programme: {course_name}\n"
            retrieved_info_str += f"Section: {chunk_name}\n\n"
        
        # Get more details about specific chunk if user wants
        print("\nWould you like more details about any of these sections? (Enter number or 'no')")
        choice = input()
        
        if choice.lower() != 'no' and choice.isdigit() and 1 <= int(choice) <= len(retrieved_results):
            idx = int(choice) - 1
            selected_chunk = retrieved_results[idx]
            course_name = selected_chunk["course_name"]
            chunk_name = selected_chunk["chunk_name"]
            chunk_content = selected_chunk["chunk_content"]
            
            print(f"\nDetails for {course_name} - {chunk_name}:")
            print(chunk_content)
            
            # Suggest related chunks
            readable_chunk_name = chunk_name.replace("_", " ").title()
            print(f"\nYou can ask more specific questions about {readable_chunk_name} for {course_name}.")
        
        # Update conversation history for context
        update_conversation_history(user_query, retrieved_info_str)
        
if __name__ == "__main__":
    # Load chunked embeddings from JSON file
    with open("data/chunked_programme_embeddings.json", "r", encoding="utf-8") as f:
        chunked_embeddings = json.load(f)
    
    # Optionally load the original processed programmes for content if needed
    try:
        with open("data/processed_programmes.json", "r", encoding="utf-8") as f:
            processed_programmes = json.load(f)
    except FileNotFoundError:
        processed_programmes = None
        print("Merged programmes file not found. Will proceed without chunk content.")
    
    # Uncomment to import chunks to Weaviate
    # import_chunks_to_weaviate(chunked_embeddings, processed_programmes)
    
    interactive_chat()
    weaviate_client.close()