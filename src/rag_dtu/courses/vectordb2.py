import weaviate
import os
import json
import re
import pickle
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from text_normalization import normalize_query
from rich.console import Console
from rich.markdown import Markdown
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import TargetVectors, MetadataQuery, Filter
import uuid

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
weaviate_api_key = "YVldI0WBz6MUZVoPZA5wp5t7zalMI12jdkfm"
weaviate_url = "https://yz34awbrqlko1tvblm77g.c0.europe-west3.gcp.weaviate.cloud"


# Create client with the required grpc_port parameter
weaviate_client = weaviate.connect_to_weaviate_cloud(
cluster_url=weaviate_url,  # Replace with your Weaviate Cloud URL
auth_credentials=Auth.api_key(weaviate_api_key),  # Replace with your Weaviate Cloud key
headers={'X-OpenAI-Api-key': os.getenv("OPENAI_API_KEY")}  # Replace with your OpenAI API key
)

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

# --- Weaviate Schema and Database Setup ---
def create_weaviate_schema():
    """
    Create Weaviate schema for courses if it doesn't exist
    """
    # Check if collection exists
    collections = weaviate_client.collections.list_all()

    weaviate_client.collections.delete("Course") 

    for collection in collections:
        if collection == "Course":
            return True
    
    # Define the collection
    course_collection = weaviate_client.collections.create(
        name="Course",
        description="A university course with detailed information",
        vectorizer_config=[
            Configure.NamedVectors.none(name = "content_embedding"),
            Configure.NamedVectors.none(name = "name_embedding")
        ],  # We'll provide our own vectors
        properties=[
            Property(
                name="course_code",
                data_type=DataType.TEXT,
                description="The 5-digit course code"
            ),
            Property(
                name="course_name",
                data_type=DataType.TEXT,
                description="The full name of the course"
            ),
            Property(
                name="content",
                data_type=DataType.TEXT,
                description="Detailed preprocessed course content"
            ),
            Property(
                name="semester",
                data_type=DataType.TEXT,
                description="Semester when course is offered"
            )
            ,
            Property(
                name="schedule",
                data_type=DataType.TEXT,
                description="Planned lectures"
            ),
            Property(
                name="ects",
                data_type=DataType.TEXT,
                description="Number of ECTS of the course"
            ),
            Property(
                name="exam",
                data_type=DataType.TEXT,
                description="Dtes of exam of the course"
            ),
            Property(
                name="signups",
                data_type=DataType.TEXT,
                description="Number of signups for the course"
            ),
            Property(
                name="average_grade",
                data_type=DataType.TEXT,
                description="Average grade of the course"
            ),
            Property(
                name="failed_students_in_percent",
                data_type=DataType.TEXT,
                description="Percentage of students who failed the course"
            ),
            Property(
                name="workload_burden",
                data_type=DataType.TEXT,
                description="Workload burden of the course"
            ),
            Property(
                name="overworked_students_in_percent",
                data_type=DataType.TEXT,
                description="Percentage of overworked students in the course"
            ),
            Property(
                name="average_rating",
                data_type=DataType.TEXT,
                description="Average rating of the course"
            )
        ]
    )
    print("Course collection created successfully")
    return True

def import_courses_to_weaviate(processed_embeddings, processed_courses):
    """
    Import processed courses into the Weaviate database using v4 client
    """
    # Make sure schema exists
    create_weaviate_schema()
    
    # Get course collection
    course_collection = weaviate_client.collections.get("Course")

    # Create list to hold course objects
    course_objs = []
    count = 0
    for course_id, course_data in processed_embeddings.items():
        count += 1
        # Get course metadata and content
        course_meta = course_data.get('metadata', {})
        content_embedding = course_data.get('content_embedding', [])
        name_embedding = course_data.get('name_embedding', [])
        content_text = processed_courses.get(course_id, {}).get('preprocessed_course', '')
        # Get embeddings
        course_name = course_meta.get('course_name', '')
        
        semester = course_meta.get('semester', '')
        semester_str = ", ".join(semester) if isinstance(semester, list) else str(semester)
        exam = course_meta.get('exam', '')
        ects = str(course_meta.get('ects', ''))
        schedule = course_meta.get('schedule', '')
        signups = str(course_meta.get('signups', ''))
        average_grade = str(course_meta.get('average_grade', ''))
        failed_students = str(course_meta.get('failed_students_in_percent', ''))
        workload_burden = str(course_meta.get('workload_burden', ''))
        overworked_students = str(course_meta.get('overworked_students_in_percent', ''))
        average_rating = str(course_meta.get('average_rating', ''))

        proper_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, course_id))
        
        # Create data object
        course_obj = weaviate.classes.data.DataObject(
            properties={
                "course_code": str(course_meta.get('course_code', '')),
                "course_name": course_name,
                "content": content_text,
                "semester": semester_str,
                "exam": exam,
                "schedule": schedule,
                "ects": ects,
                "signups": signups,
                "average_grade": average_grade,
                "failed_students_in_percent": failed_students,
                "workload_burden": workload_burden,
                "overworked_students_in_percent": overworked_students,
                "average_rating": average_rating
            },
            uuid=proper_uuid,
            vector= {
                "content_embedding": content_embedding,
                "name_embedding": name_embedding
            }  # Make sure this matches your schema's vector name
        )
        print(f"Adding course {course_id} to Weaviate")
        course_objs.append(course_obj)
    for obj in course_objs:
        try:
            course_collection.data.insert(
                properties=obj.properties,
                uuid=obj.uuid,
                vector=obj.vector
            )
            print(f"✅ Inserted course {obj.properties.get('course_code')}")
        except Exception as e:
            print(f"❌ Error inserting course {obj.properties.get('course_code')}: {e}")


    print(f"Imported {len(processed_courses)} courses to Weaviate")
def check_weaviate_courses():
    """
    Check if courses are already in Weaviate using v4 client
    """
    try:
        collections = weaviate_client.collections.list_all()
        for collection in collections:
            if collection == "Course":
                print("Course collection already exists")
            
        # Check if collection has data
        course_collection = weaviate_client.collections.get("Course")
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
def search_courses(query, top_k=3, filters=None, weight_content=0.7, weight_name=0.3):
    """
    Search for courses using Weaviate with hybrid search (name and content)
    """
    course_collection = weaviate_client.collections.get("Course")

    weight_content = weight_content * 100
    weight_name = weight_name * 100
    
    # First: Exact lookup based on course code
    if filters and 'course_code' in filters:
        try:
            response = course_collection.query.fetch_objects(
                filters=Filter.by_property("course_code").equal(filters['course_code']),
            )
            
            if len(response.objects) > 0:
                return [obj.properties for obj in response.objects]
        except Exception as e:
            print(f"Error during course code search: {e}")
    
    # Get embeddings for the query
    query_embedding_openai = get_embedding(query)
    name_embedding_trans = get_name_embedding(query)
    
    # Perform hybrid search using both vectors with weights
    try:
        response = course_collection.query.hybrid(
            query =query,
            alpha = 0.75,
            vector={
                "content_embedding": query_embedding_openai,
                "name_embedding": name_embedding_trans
            },
            limit=top_k,
            target_vector=TargetVectors.manual_weights({"content_embedding": weight_content ,
                                                         "name_embedding" : weight_name}),
            return_metadata=MetadataQuery(distance=True)
        )

        if len(response.objects) > 0:
            # Sort by distance
            return [obj.properties for obj in response.objects]
    except Exception as e:
        print(f"Error during hybrid search: {e}")

    return []


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
    for course in courses_info:
        course_code = course.get("course_code", "N/A")
        course_name = course.get("course_name", "Unnamed Course")
        preprocessed_text = course.get("content", "")
        semester = course.get("semester", "N/A"),
        ects = course.get("ects", "N/A"),
        schedule = course.get("schedule", "N/A"),
        exam = course.get("exam", "N/A"),
        signups = course.get("signups", "N/A"),
        average_grade = course.get("average_grade", "N/A"),
        failed_students = course.get("failed_students_in_percent", "N/A"),
        workload_burden = course.get("workload_burden", "N/A"),
        overworked_students = course.get("overworked_students_in_percent", "N/A"),
        average_rating = course.get("average_rating", "N/A"),

        context_lines.append(
            f"Course Code: {course_code}\n"
            f"Title: {course_name}\n"
            f"ECTS: {ects}\n"
            f"Semester: {semester}\n"
            f"Schedule: {schedule}\n"
            f"Exam Date: {exam}\n"
            f"Details: {preprocessed_text}\n"
            f"Signups: {signups}\n"
            f"Average Grade: {average_grade}\n"
            f"Failed Students: {failed_students}\n"
            f"Workload Burden: {workload_burden}\n"
            f"Overworked Students: {overworked_students}\n"
            f"Average Rating: {average_rating}\n"
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
def interactive_chat(processed_embedding_path, processed_courses_path):
    print("Starting interactive chat session. Type 'exit' or 'quit' to end.")
    print("\n 🧠 Assistant 🧠: Hi! I am your assistant for DTU courses information :) How can I help you?")
    
    # Load processed courses if needed for additional information
    with open(processed_courses_path, "r", encoding="utf-8") as f:
        processed_courses = json.load(f)
    
    with open(processed_embedding_path, "r", encoding="utf-8") as f:
        processed_embeddings = json.load(f)
    
    # Check if courses are in Weaviate, import if not
    if not check_weaviate_courses():
        print("Importing courses to Weaviate...")
        import_courses_to_weaviate(processed_embeddings, processed_courses)
        print("Courses imported successfully.")
    
    while True:
        user_query = input("\n 👨‍💼 User 👨‍💼: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            print("Exiting interactive chat.")
            print("Goodbye!")
            break

        normalized_query, _ = normalize_query(user_query)
        filters = extract_filters_from_query(normalized_query)

        # Search Weaviate using the user query
        search_results = search_courses(normalized_query, top_k=5, filters=filters)


        # Format retrieved courses for conversation history
        retrieved_courses_str = ""
        for course in search_results:
            course_code = course.get("course_code", "N/A")
            course_name = course.get("course_name", "Unnamed Course")
            content = course.get("content", "No content available")
            semester = course.get("semester", "N/A")
            ects = course.get("ects", "N/A")
            schedule = course.get("schedule", "N/A")
            exam = course.get("exam", "N/A")
            signups = course.get("signups", "N/A")
            average_grade = course.get("average_grade", "N/A")
            failed_students = course.get("failed_students_in_percent", "N/A")
            workload_burden = course.get("workload_burden", "N/A")
            overworked_students = course.get("overworked_students_in_percent", "N/A")
            average_rating = course.get("average_rating", "N/A")

            # Format the course information
            retrieved_courses_str += f"Course Code: {course_code}\n"
            retrieved_courses_str += f"Title: {course_name}\n"
            retrieved_courses_str += f"ECTS: {ects}\n"
            retrieved_courses_str += f"Semester: {semester}\n"
            retrieved_courses_str += f"Schedule: {schedule}\n"
            retrieved_courses_str += f"Exam Date: {exam}\n"
            retrieved_courses_str += f"Signups: {signups}\n"
            retrieved_courses_str += f"Average Grade: {average_grade}\n"
            retrieved_courses_str += f"Failed Students: {failed_students}\n"
            retrieved_courses_str += f"Workload Burden: {workload_burden}\n"
            retrieved_courses_str += f"Overworked Students: {overworked_students}\n"
            retrieved_courses_str += f"Average Rating: {average_rating}\n"
            retrieved_courses_str += f"Details: {content}\n\n"
            retrieved_courses_str += "---\n\n"
        
        # Get GPT-4 answer along with the current prompt (for logging purposes)
        answer, _ = query_with_gpt4(user_query, search_results)
        print("\n 🧠 Assistant 🧠:")
        display_markdown(answer)
        print("\n---\n")

        # Update the conversation history with the user query and the retrieved course summary
        update_conversation_history(user_query, retrieved_courses_str)

    weaviate_client.close()


# --- Main Flow ---
if __name__ == "__main__":
    processed_embedding_path = "data/course_embeddings.json"
    processed_courses_path = "data/processed_courses.json"

    with open(processed_embedding_path, "r", encoding="utf-8") as f:
        processed_embeddings = json.load(f)
    
    print("Loaded processed embeddings.")
    #print first embedding
    
    # Start the interactive chat session
    interactive_chat(processed_embedding_path, processed_courses_path)