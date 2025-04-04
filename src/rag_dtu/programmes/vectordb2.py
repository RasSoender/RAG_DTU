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
        raise ValueError(f"❌ Errore: input non valido per embedding: {repr(text)}")
    
    print(f"📤 Richiesta embedding per: {repr(text)}")
    response = client.embeddings.create(input=[text], model=model)  # Nota: input deve essere una lista
    return response.data[0].embedding

# Initialize a specialized model for short texts (course names)
name_embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_name_embedding(text):
    """
    Returns a 384-dimensional embedding for short text like course names.
    """
    embedding = name_embedding_model.encode(text)
    return embedding.tolist()

# --- Weaviate Schema and Database Setup ---
def create_weaviate_schema():
    """
    Create Weaviate schema for courses if it doesn't exist
    """
    # Check if collection exists
    collections = weaviate_client.collections.list_all()

    print("Existing collections:", collections)

    for collection in collections:
        if collection == "Programmes":
            return True
    
    # Define the collection
    course_collection = weaviate_client.collections.create(
        name="Programmes",
        description="A master's programme at DTU",
        vectorizer_config=[
        Configure.NamedVectors.none(name="name_embedding"),
        Configure.NamedVectors.none(name="programme_specification"),
        Configure.NamedVectors.none(name="duration"),
        Configure.NamedVectors.none(name="admission_requirements"),
        Configure.NamedVectors.none(name="academic_requirements"),
        Configure.NamedVectors.none(name="competence_profile"),
        Configure.NamedVectors.none(name="learning_objectives"),
        Configure.NamedVectors.none(name="structure"),
        Configure.NamedVectors.none(name="programme_provision"),
        Configure.NamedVectors.none(name="specializations"),
        Configure.NamedVectors.none(name="curriculum"),
        Configure.NamedVectors.none(name="master_thesis"),
        Configure.NamedVectors.none(name="master_thesis_specific_rules"),
        Configure.NamedVectors.none(name="activity_requirements"),
        Configure.NamedVectors.none(name="programme_rules"),
        Configure.NamedVectors.none(name="course_descriptions"),
        Configure.NamedVectors.none(name="course_registration"),
        Configure.NamedVectors.none(name="binding_courses"),
        Configure.NamedVectors.none(name="academic_prerequisites"),
        Configure.NamedVectors.none(name="limited_admission_courses"),
        Configure.NamedVectors.none(name="mandatory_participation"),
        Configure.NamedVectors.none(name="teaching_material_deadlines"),
        Configure.NamedVectors.none(name="project_courses"),
        Configure.NamedVectors.none(name="evaluation_of_teaching"),
        Configure.NamedVectors.none(name="exam_rules"),
        Configure.NamedVectors.none(name="other_info"),
        Configure.NamedVectors.none(name="head_of_study")
    ],  # We'll provide our own vectors
        properties=[
            Property(
                name="course_name",
                data_type=DataType.TEXT,
                description="The full name of the course"
            )

        ]
    )
    print("Course collection created successfully")
    return True

def import_courses_to_weaviate(processed_embeddings):
    """
    Import processed courses into the Weaviate database using v4 client
    """
    # Make sure schema exists
    create_weaviate_schema()
    
    check_weaviate_courses()
    # Get course collection
    course_collection = weaviate_client.collections.get("Programmes")

    # Create list to hold course objects
    course_objs = []
    count = 0
    for course_name, course_data in processed_embeddings.items():
        print(f"Processing course {count + 1}: {course_name}")
        count += 1
        # Get course metadata and content
        course_meta = course_data.get('metadata', {})
        name_embedding = course_data.get('name_embedding', [])
        programme_specification= course_data.get('programme_specification', [])
        duration= course_data.get('duration', [])
        admission_requirements = course_data.get('admission_requirements', [])
        academic_requirements = course_data.get('academic_requirements', [])
        competence_profile = course_data.get('competence_profile', [])
        learning_objectives = course_data.get('learning_objectives', [])
        structure = course_data.get('structure', [])
        programme_provision = course_data.get('programme_provision', [])
        specializations = course_data.get('specializations', [])
        curriculum = course_data.get('curriculum', [])
        master_thesis = course_data.get('master_thesis', [])
        master_thesis_specific_rules = course_data.get('master_thesis_specific_rules', [])
        activity_requirements = course_data.get('activity_requirements', [])
        programme_rules = course_data.get('programme_rules', [])
        course_descriptions          = course_data.get('course_descriptions', [])
        course_registration          = course_data.get('course_registration', [])
        binding_courses              = course_data.get('binding_courses', [])
        academic_prerequisites       = course_data.get('academic_prerequisites', [])
        limited_admission_courses    = course_data.get('limited_admission_courses', [])
        mandatory_participation      = course_data.get('mandatory_participation', [])
        teaching_material_deadlines  = course_data.get('teaching_material_deadlines', [])
        project_courses              = course_data.get('project_courses', [])
        evaluation_of_teaching       = course_data.get('evaluation_of_teaching', [])
        exam_rules                   = course_data.get('exam_rules', [])
        other_info                   = course_data.get('other_info', [])
        head_of_study                = course_data.get('head_of_study', [])


        # Get embeddings
        course_name_extended = course_meta.get('course_name', '')

        print(f"Course name: {course_name_extended}")
        
        proper_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, course_name))
        
        # Create data object
        course_obj = weaviate.classes.data.DataObject(
            properties={
                "course_name": course_name,
            },
            uuid=proper_uuid,
            vector= {
                "name_embedding": name_embedding,
                "programme_specification": programme_specification,
                "duration": duration,
                "admission_requirements": admission_requirements,
                "academic_requirements": academic_requirements,
                "competence_profile": competence_profile,
                "learning_objectives": learning_objectives,
                "structure": structure,
                "programme_provision": programme_provision,
                "specializations": specializations,
                "curriculum": curriculum,
                "master_thesis": master_thesis,
                "master_thesis_specific_rules": master_thesis_specific_rules,
                "activity_requirements": activity_requirements,
                "programme_rules": programme_rules,
                "course_descriptions": course_descriptions,
                "course_registration": course_registration,
                "binding_courses": binding_courses, 
                "academic_prerequisites": academic_prerequisites,
                "limited_admission_courses": limited_admission_courses,
                "mandatory_participation": mandatory_participation,
                "teaching_material_deadlines": teaching_material_deadlines,
                "project_courses": project_courses,
                "evaluation_of_teaching": evaluation_of_teaching,
                "exam_rules": exam_rules,
                "other_info": other_info,
                "head_of_study": head_of_study
            }  # Make sure this matches your schema's vector name
        )
        print(f"Adding course {course_name} to Weaviate")
        course_objs.append(course_obj)
    
    # Insert all courses into Weaviate
    for obj in course_objs:
        try:
            course_collection.data.insert(
                properties=obj.properties,
                uuid=obj.uuid,
                vector=obj.vector
            )
            print(f"✅ Inserted course {obj.properties.get('course_name')}")
        except Exception as e:
            print(f"❌ Error inserting course {obj.properties.get('course_name')}: {e}")
    
    # Fixed: Use the length of course_objs instead of undefined processed_courses
    print(f"Imported {len(course_objs)} courses to Weaviate")

def check_weaviate_courses():
    """
    Check if courses are already in Weaviate using v4 client
    """
    try:
        collections = weaviate_client.collections.list_all()
        for collection in collections:
            if collection == "Programmes":
                print("Programmes collection already exists")
            
        # Check if collection has data
        course_collection = weaviate_client.collections.get("Programmes")
        response = course_collection.query.fetch_objects(
            limit=1
        )
        
        if len(response.objects) > 0:
            return True
        return False
    except Exception as e:
        print(f"Error checking Weaviate courses: {e}")
        return False

# --- Weaviate Query Functions ---
def search_courses(query, top_k=3):
    """
    Search for courses using Weaviate with hybrid search across all vector fields
    """
    course_collection = weaviate_client.collections.get("Programmes")
    
    # Get embeddings for the query
    query_embedding_openai = get_embedding(query)
    name_embedding_trans = get_name_embedding(query)
    
    # Create vector query for all vector fields
    vector_query = {
        "name_embedding": name_embedding_trans,
        "programme_specification": query_embedding_openai,
        "duration": query_embedding_openai,
        "admission_requirements": query_embedding_openai,
        "academic_requirements": query_embedding_openai,
        "competence_profile": query_embedding_openai,
        "learning_objectives": query_embedding_openai,
        "structure": query_embedding_openai,
        "programme_provision": query_embedding_openai,
        "specializations": query_embedding_openai,
        "curriculum": query_embedding_openai,
        "master_thesis": query_embedding_openai,
        "master_thesis_specific_rules": query_embedding_openai,
        "activity_requirements": query_embedding_openai,
        "programme_rules": query_embedding_openai,
        "course_descriptions": query_embedding_openai,
        "course_registration": query_embedding_openai,
        "binding_courses": query_embedding_openai,
        "academic_prerequisites": query_embedding_openai,
        "limited_admission_courses": query_embedding_openai,
        "mandatory_participation": query_embedding_openai,
        "teaching_material_deadlines": query_embedding_openai,
        "project_courses": query_embedding_openai,
        "evaluation_of_teaching": query_embedding_openai,
        "exam_rules": query_embedding_openai,
        "other_info": query_embedding_openai,
        "head_of_study": query_embedding_openai
    }
    
    # Define target vectors with equal weights
    # Distribute weight across all vectors with slight emphasis on name_embedding
    vector_count = len(vector_query)
    name_weight = 20  # Give more weight to the name
    regular_weight = (100 - name_weight) / (vector_count - 1)  # Distribute remaining weight
    
    weights = {field: regular_weight for field in vector_query.keys()}
    weights["name_embedding"] = name_weight  # Override with higher weight for name
    
    # Perform hybrid search using all vectors
    try:
        response = course_collection.query.hybrid(
            query=query,
            alpha=0.75,  # Blend between vector and keyword search
            vector=vector_query,
            limit=top_k,
            target_vector=TargetVectors.manual_weights(weights),
            return_metadata=MetadataQuery(distance=True)
        )

        if len(response.objects) > 0:
            return [obj for obj in response.objects]
    except Exception as e:
        print(f"Error during hybrid search: {e}")
        # Try a simpler approach if there's an error
        try:
            print("Trying simplified search...")
            response = course_collection.query.hybrid(
                query=query,
                alpha=0.75,
                vector={"name_embedding": name_embedding_trans},
                limit=top_k,
                return_metadata=MetadataQuery(distance=True)
            )
            if len(response.objects) > 0:
                return [obj.properties for obj in response.objects]
        except Exception as e2:
            print(f"Error during simplified search: {e2}")

    return []



def interactive_chat():
    """
    Run an interactive chat session that allows testing the course retrieval system.
    """
    print("Welcome to the DTU Course Assistant!")
    print("Ask me anything about DTU master's programmes. Type 'exit' to quit.\n")
    
    # Check if courses exist in Weaviate
    if not check_weaviate_courses():
        print("No courses found in Weaviate. Please make sure courses are imported first.")
        return
    else:
        print("Courses found in Weaviate, ready to query.")
    
    while True:
        # Get user query
        user_query = "How does it work the master thesis for the master's programme in Human Centered AI?"
        
        # Exit condition
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Thanks for chatting! Goodbye!")
            break
        
        # Normalize the query
        normalized_query, _ = normalize_query(user_query)
        
        # Search for relevant courses
        print("Searching for relevant programmes...")
        retrieved_courses = search_courses(normalized_query, top_k=3)
        
        if not retrieved_courses:
            print("Sorry, I couldn't find any relevant programmes.")
            continue
        
        # Format retrieved information
        
        print(retrieved_courses)
        
        # Get more details about specific course if user wants
        print("\nWould you like more details about any of these programmes? (Enter number or 'no')")
        choice = input()
        
        if choice.lower() != 'no' and choice.isdigit() and 1 <= int(choice) <= len(retrieved_courses):
            idx = int(choice) - 1
            selected_course = retrieved_courses[idx]
            course_name = selected_course.get("course_name", "Unknown Programme")
            
            # Simple message with programme details
            message = f"Details for {course_name}:\n"
            message += f"This is a master's programme at DTU."
            
            print(message)
        
        # Update conversation history for context
        update_conversation_history(user_query, retrieved_info_str)

if __name__ == "__main__":
    # Load processed courses from JSON file
    with open("data/programme_embeddings.json", "r", encoding="utf-8") as f:
        processed_courses = json.load(f)
    
    # Import courses to Weaviate
    import_courses_to_weaviate(processed_courses)

    interactive_chat()
    weaviate_client.close()
    
