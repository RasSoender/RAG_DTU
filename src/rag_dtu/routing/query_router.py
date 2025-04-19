import weaviate
import os
import json
import re
import uuid
import time
import difflib
from typing import List, Dict, Any, Optional, Tuple
from rag_dtu.programmes.text_normalization import normalize_query
from dataclasses import dataclass
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.markdown import Markdown
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import TargetVectors, MetadataQuery, Filter
from dotenv import load_dotenv
load_dotenv()

# Constants and Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
# LLM_MODEL = "gpt-4o-mini"
LLM_MODEL = "gpt-4.1-mini"

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
console = Console()

# Initialize embedding models
name_embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define database configurations
vector_db_configs = {
    "programme_db": {
        "url": "https://yz34awbrqlko1tvblm77g.c0.europe-west3.gcp.weaviate.cloud",
        "api_key": "YVldI0WBz6MUZVoPZA5wp5t7zalMI12jdkfm",
        "collection_name": "Chunk"  # Programme collection
    },
    "course_db": {
        "url": "https://yz34awbrqlko1tvblm77g.c0.europe-west3.gcp.weaviate.cloud",
        "api_key": "YVldI0WBz6MUZVoPZA5wp5t7zalMI12jdkfm",  # Replace with your second API key
        "collection_name": "Course"
    }
}

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

@dataclass
class MemoryItem:
    """A structured memory item for the conversation history."""
    timestamp: float
    user_query: str
    query_type: str
    retrieved_info: str
    used_before_info : str = None
    metadata: Dict[str, Any] = None

@dataclass
class QueryMemoryItem:
    """A structured memory item for the conversation history."""
    timestamp: float
    user_query: str
    query_type: str
    additional_info: str
    used_before_info : str = None


class QueryMemory:
    """A simple memory system to store conversation history."""
    def __init__(self, max_items: int = 10):
        self.memory: List[QueryMemoryItem] = []
        self.max_items = max_items
    
    def add(self, user_query: str, query_type: str, 
            additional_info: str,used_before_info: str = None, 
            metadata: Dict[str, Any] = None) -> None:
        """Add a new item to memory with current timestamp."""
        memory_item = QueryMemoryItem(
            timestamp=time.time(),
            user_query=user_query,
            query_type=query_type,
            additional_info=additional_info,
            used_before_info=used_before_info
        )
        self.memory.append(memory_item)
        # Trim memory if it exceeds max size
        if len(self.memory) > self.max_items:
            self.memory = self.memory[-self.max_items:]
    def get_last_n_items(self, n: int) -> List[QueryMemoryItem]:
        """Get the last n items from memory."""
        return self.memory[-n:] if n <= len(self.memory) else self.memory
    def get_formatted_history(self, relevant_items: List[QueryMemoryItem] = None) -> str:
        """Format memory items for inclusion in prompts."""
        items_to_format = relevant_items if relevant_items is not None else self.memory
        if not items_to_format:
            return "No conversation history available."
        history_parts = []
        for i, item in enumerate(items_to_format):
            # Format simplified retrieved info for prompt context
            history_parts.append(
                f"Exchange {i+1}:\n"
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item.timestamp))}\n"
                f"User: {item.user_query}\n"
                f"Type: {item.query_type}\n"
                f"Used Before Information: {item.used_before_info}\n"
                f"Additional Information:\n{item.additional_info}\n"
            )
        return "\n---\n".join(history_parts)
    def clear(self) -> None:
        """Clear all memory items."""
        self.memory = []

class Memory:
    """
    Enhanced short-term memory system with advanced querying capabilities.
    (Note: The new implementation only stores items and formats the entire conversation history.)
    """
    def __init__(self, max_items: int = 10):
        self.memory: List[MemoryItem] = []
        self.max_items = max_items
    
    def add(self, user_query: str, query_type: str, 
            retrieved_info: str, used_before_info: str = None,
            metadata: Dict[str, Any] = None) -> None:
        """Add a new item to memory with current timestamp."""
        memory_item = MemoryItem(
            timestamp=time.time(),
            user_query=user_query,
            query_type=query_type,
            retrieved_info=retrieved_info,
            used_before_info=used_before_info,
            metadata=metadata
        )
        self.memory.append(memory_item)
        # Trim memory if it exceeds max size
        if len(self.memory) > self.max_items:
            self.memory = self.memory[-self.max_items:]

    def get_last_n_items(self, n: int) -> List[MemoryItem]:
        """Get the last n items from memory."""
        return self.memory[-n:] if n <= len(self.memory) else self.memory
    
    def get_formatted_history(self, relevant_items: List[MemoryItem] = None) -> str:
        """Format memory items for inclusion in prompts."""
        items_to_format = relevant_items if relevant_items is not None else self.memory
        if not items_to_format:
            return "No conversation history available."
        history_parts = []
        for i, item in enumerate(items_to_format):
            # Format simplified retrieved info for prompt context
            history_parts.append(
                f"Exchange {i+1}:\n"
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item.timestamp))}\n"
                f"User: {item.user_query}\n"
                f"Type: {item.query_type}\n"
                f"Used Before Information: {item.used_before_info}\n"
                f"Retrieved Information:\n{item.retrieved_info}\n"
                f"Metadata: {json.dumps(item.metadata, indent=2) if item.metadata else 'None'}"
            )
        return "\n---\n".join(history_parts)
    
    def clear(self) -> None:
        """Clear all memory items."""
        self.memory = []

class MultiVectorDBClient:
    """Wrapper for interacting with multiple vector databases."""
    def __init__(self, vector_db_configs: Dict[str, Dict[str, str]]):
        self.configs = vector_db_configs
        self.clients = {}
        self._connect_all()
        
    def _connect_all(self):
        """Initialize connections to all configured databases."""
        for db_name, config in self.configs.items():
            try:
                self.clients[db_name] = self._connect_to_db(config["url"], config["api_key"])
                print(f"Connected to {db_name} at {config['url']}")
            except Exception as e:
                print(f"Error connecting to {db_name}: {e}")
                self.clients[db_name] = None
    
    def _connect_to_db(self, url: str, api_key: str):
        """Connect to a specific Weaviate instance."""
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
            headers={'X-OpenAI-Api-key': OPENAI_API_KEY}
        )
    
    def is_ready(self, db_name: str = None) -> bool:
        """Check if a specific or all databases are ready."""
        if db_name:
            client = self.clients.get(db_name)
            return client.is_ready() if client else False
        
        # Check all clients
        return all(client.is_ready() for client in self.clients.values() if client)
    
    def close(self) -> None:
        """Close all database connections."""
        for client in self.clients.values():
            if client:
                client.close()
    
    def get_available_programmes(self) -> List[str]:
        """Get list of all available programmes in the database."""
        programme_db = "programme_db"
        if programme_db not in self.clients or not self.clients[programme_db]:
            print(f"Programme database not connected")
            return []
            
        try:
            programme_collection = self.clients[programme_db].collections.get(
                self.configs[programme_db]["collection_name"]
            )
            print(f"Fetching available programmes from collection: {self.configs[programme_db]['collection_name']}")
            
            # Fetch all objects from the collection
            print("Fetching all objects from the collection...")
            response = programme_collection.query.fetch_objects(
                limit=10000,
                return_properties=["course_name"]  # "course_name" is the programme name
            )
            programmes = set()
            for obj in response.objects:
                if "course_name" in obj.properties:
                    programmes.add(obj.properties["course_name"])
            return sorted(list(programmes))
        except Exception as e:
            print(f"Error fetching available programmes: {e}")
            return []
    
    def search_courses(self, query: str, top_k: int = 7, filters: Dict = None) -> List[Dict]:
        """Search for courses using hybrid search."""
        course_db = "course_db"
        if course_db not in self.clients or not self.clients[course_db]:
            print(f"Course database not connected")
            return []
            
        try:
            course_collection = self.clients[course_db].collections.get(
                self.configs[course_db]["collection_name"]
            )
            
            # Check for direct course code lookup first
            if filters and 'course_code' in filters:
                response = course_collection.query.fetch_objects(
                    filters=Filter.by_property("course_code").equal(filters['course_code']),
                )
                if len(response.objects) > 0:
                    return [obj.properties for obj in response.objects]
                    
            # Get embeddings for query
            content_embedding = get_embedding(query)
            name_embedding = get_name_embedding(query)
            
            # Perform hybrid search
            response = course_collection.query.hybrid(
                query=query,
                alpha=0.75,
                vector={
                    "content_embedding": content_embedding,
                    "name_embedding": name_embedding
                },
                limit=top_k,
                target_vector=TargetVectors.manual_weights({
                    "content_embedding": 70,
                    "name_embedding": 30
                }),
                return_metadata=MetadataQuery(distance=True)
            )
            
            if response.objects:
                return [obj.properties for obj in response.objects]
            return []
        except Exception as e:
            print(f"Error in search_courses: {e}")
            return []
        
    def build_context_str_courses(self, search_results: List[Dict]) -> str:
        context_lines = []
        query_information = ""
        context_information = ""
        for course in search_results:
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
            url = course.get("url", "N/A")

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
                f"URL: {url}\n"
            )

            # Format the course information
            context_information += f"Course Code: {course_code}\n"
            context_information += f"Title: {course_name}\n"
            context_information += f"ECTS: {ects}\n"
            context_information += f"Semester: {semester}\n"
            context_information += f"Schedule: {schedule}\n"
            context_information += f"Exam Date: {exam}\n"
            context_information += f"Signups: {signups}\n"
            context_information += f"Average Grade: {average_grade}\n"
            context_information += f"Failed Students: {failed_students}\n"
            context_information += f"Workload Burden: {workload_burden}\n"
            context_information += f"Overworked Students: {overworked_students}\n"
            context_information += f"Average Rating: {average_rating}\n"
            context_information += f"URL: {url}\n"
            context_information += f"Details: {preprocessed_text}\n\n"
            context_information += "---\n\n"

            query_information += f"Course Code: {course_code}\n"
            query_information += f"Title: {course_name}\n"
            context_information += "---\n\n"

        context_str = "\n\n---\n\n".join(context_lines)
        return context_str, query_information, context_information
    
    def search_programmes(self, query: str, top_k: int = 5, filter_programme: str = None) -> List[Dict]:
        """Search for programme information."""
        programme_db = "programme_db"
        if programme_db not in self.clients or not self.clients[programme_db]:
            print(f"Programme database not connected")
            return []
            
        try:
            programme_collection = self.clients[programme_db].collections.get(
                self.configs[programme_db]["collection_name"]
            )
            
            # Get embeddings for query
            content_embedding = get_embedding(query)
            
            name_embedding = get_name_embedding(query)
            
            # Create filter if programme name is provided
            filter_obj = None
            if filter_programme:
                filter_obj = Filter.by_property("course_name").equal(filter_programme)
                
            # Perform hybrid search
            response = programme_collection.query.hybrid(
                query=query,
                alpha=0.75,
                vector={
                    "content_embedding": content_embedding,
                    "name_embedding": name_embedding
                },
                limit=top_k,
                target_vector=TargetVectors.manual_weights({
                    "content_embedding": 70,
                    "name_embedding": 30
                }),
                return_metadata=MetadataQuery(distance=True),
                return_properties=["course_name", "chunk_name", "chunk_content", "chunk_url"],
                filters=filter_obj
            )
            
            results = []
            for obj in response.objects:
                results.append({
                    "programme_name": obj.properties.get("course_name", "Unknown Programme"),
                    "section_name": obj.properties.get("chunk_name", "Unknown Section"),
                    "content": obj.properties.get("chunk_content", ""),
                    "chunk_url": obj.properties.get("chunk_url", ""),
                    "distance": obj.metadata.distance if hasattr(obj, 'metadata') else None
                })
            return results
        except Exception as e:
            print(f"Error in search_programmes: {e}")
            return []

    def build_context_str_programmes(self, search_results: List[Dict]) -> str:
        context_lines = []
        query_information = ""
        context_information = ""
        for information in search_results:
            course_name = information.get("programme_name", "N/A")
            chunk_name = information.get("section_name", "Unnamed Course")
            chunk_content = information.get("content", "")
            chunk_url = information.get("chunk_url", "N/A")

            context_lines.append(
                f"Master's programme name: {course_name}\n"
                f"Chunk name: {chunk_name}\n"
                f"Chunk content: {chunk_content}\n"
                f"Chunk URL: {chunk_url}\n"
            )

            query_information += f"Master's programme name: {course_name}\n"
            query_information += f"Chunk name: {chunk_name}\n"
            query_information += "---\n\n"

            context_information += f"Master's programme name: {course_name}\n"
            context_information += f"Chunk name: {chunk_name}\n"
            context_information += f"Chunk content: {chunk_content}\n"
            context_information += f"Chunk URL: {chunk_url}\n"
            context_information += "---\n\n"

        context_str = "\n\n---\n\n".join(context_lines)
        return context_str, query_information, context_information
    
class QueryRouter:
    """
    Router that analyzes user queries, determines intent type and if history is required.
    It then routes the query to perform either a search or a conversation.
    The LLM is provided with a list of available programme names so that if the query
    is about a programme, it outputs an exact matching programme name.
    """
    QUERY_TYPES = {
        "COURSE": "course_query",
        "PROGRAMME": "programme_query",
        "CONVERSATION": "conversation_query",
        "UNKNOWN": "unknown_query"
    }
    
    def __init__(self, query_memory: QueryMemory, context_memory: Memory, vector_db_client: MultiVectorDBClient):
        self.query_memory = query_memory
        self.context_memory = context_memory
        self.vector_db_client = vector_db_client
        self.openai_client = openai_client
        self.previous_query_analysis = ""
        # Retrieve available programmes list from Weaviate
        self.available_programmes = self.vector_db_client.get_available_programmes()
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user query to determine intent type, whether conversation history is needed,
        and extract additional details such as course_code and programme_name if applicable.
        The LLM prompt includes the list of available programmes so that, for programme queries,
        it outputs the exact matching programme name.
        Returns a structured analysis as a JSON object.
        """
        # Get whole conversation history
        history_query = self.query_memory.get_last_n_items(5)  # Get last 5 items for context
        history_context_query = self.query_memory.get_formatted_history(history_query)
        
        available_programmes_str = ", ".join(self.available_programmes) if self.available_programmes else "None"
        prompt = f"""
You are an expert academic assistant helping with queries about university courses and programmes.
Analyze the following user query and determine what the user is asking about.

### Query:
{query}

### Recent Conversation Context:
{history_context_query}

### Available Programmes:
{available_programmes_str}

Please provide a structured analysis with the following fields:
1. query_type: Exactly one of ["course_query", "programme_query", "conversation_query", "unknown_query"]
    WARNING: If there is a code of 5 digits like 02450 it is a course_query
   - course_query: The user is asking information about a course: the information that can be useful for asserting if the question is regarding a course are like course code, course name, course content, course semester, course schedule, course exam date, course signups, course average grade, course failed students in percent, course workload burden, course overworked students in percent, and course average rating, prerequisites for a course, course requirements, course teacher etc.
   - programme_query: The user is asking about degree programmes or related information. These information could be regarding programmes like thesis, requirements,learning objective, head of study, currriculu, specializations, etc. For asserting well, always check the programme name in the query and the available programme list, if there is nothing that match close, do not use this field. Moreover, if asked about if a specific course is mandatory for a specific programme, it is a programme query.
   - conversation_query: If there is a question that make sense but the user doesn't really specify a programme name, or a course, or a course code, probably the user is referring to past conversations and so this is the correct field. For example, if the user refer to "that master" or "that course" or similar, this is the correct field. WARNING: If the questions is regarding past queries, but there is a course name, or a programme names etc, prefer other query_type with respect to this, and returns requires_history = true.
   - unknown_query: The query doesn't clearly fit the above categories.
2. requires_history: true/false, indicating if the query requires conversation history for context since the query is not specific enough. Usually, when it falls to conversation_query field, it is also required
3. course_code: If applicable, a 5-digit course code extracted from the query or, if you find that the course code is in the past queries, take it from the conversation history.
4. programme_name: If applicable but just only if there is explicitly written in the query, if the query appears to be about a programme, output the exact programme name from the provided list that most closely matches the query; if none match, output an empty string. Moreover, if you find that the newest query can be referred to a programme name of the past queries, give it.

###WARNING
1. If in the query there is the word exam, course, probably the user is referring to course_query
2. If in the query there is the word programme, master etc, probably the user is referring to programme_query

Format your response as a valid JSON object with these fields.
"""
        try:
            response = self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert academic query analyzer. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}  # For newer API versions that support this
            )
            content = response.choices[0].message.content
            # Additional safety check
            if not content.strip().startswith("{"):
                raise ValueError("Response is not valid JSON")
            analysis = json.loads(content)
            return analysis
        except Exception as e:
            print(f"Error analyzing query: {e}")
            print(f"Raw response: {response.choices[0].message.content if 'response' in locals() else 'No response'}")
            return {
                "query_type": self.QUERY_TYPES["UNKNOWN"],
                "requires_history": False,
                "course_code": "",
                "programme_name": ""
            }
    
    def route_query(self, query: str, master: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Determine the query type and route it:
        - For course queries, perform a course search.
        - For programme queries, perform a programme search using the exact programme name provided in the analysis.
        - For conversation or unknown queries, generate an LLM response.
        Returns the assistant's response along with analysis details.
        """
        print(f"Routing query: {query}")
        analysis = self.analyze_query(query)
        query_type = analysis.get("query_type", self.QUERY_TYPES["UNKNOWN"])
        print(f"Query type: {query_type}")
        print( f"{master}")
        requires_history = analysis.get("requires_history", False)
        history = self.context_memory.get_last_n_items(5) if requires_history else []
        history_context = (
            self.context_memory.get_formatted_history(history)
            if history else "No conversation history available."
        )
        answer = ""
        other_info = None
        query_information = ""
        context_information = ""
        if query_type == self.QUERY_TYPES["COURSE"]:
            filters = extract_filters_from_query(query)
            normalized_query, _ = normalize_query(query)
            search_results = self.vector_db_client.search_courses(normalized_query, filters=filters)
            if search_results:
                context_str, query_information, context_information = self.vector_db_client.build_context_str_courses(search_results)
                response = self._generate_response(query, query_type, requires_history, history_context, context_str)
                answer = response.get("response", "")
                other_info = response.get("other", None)
                if other_info:
                    context_information = [result for result in search_results if result.get("course_code") == other_info]
            self.previous_query_analysis = analysis
        elif query_type == self.QUERY_TYPES["PROGRAMME"]:
            prog_name = analysis.get("programme_name", "").strip()
            normalized_query, _ = normalize_query(query)
            if master:
                prog_name = master
            search_results = self.vector_db_client.search_programmes(query, filter_programme=prog_name if prog_name else None)
            # print(f"Search results: {search_results}")
            if search_results:
                context_str, query_information, context_information = self.vector_db_client.build_context_str_programmes(search_results)
                response = self._generate_response(query, query_type, requires_history, history_context, context_str)
                answer = response.get("response", "")
                other_info = response.get("other", None)
            self.previous_query_analysis = analysis
        elif query_type == self.QUERY_TYPES["CONVERSATION"]:
            context_str = ""
            previous_query_type = self.previous_query_analysis.get("query_type", self.QUERY_TYPES["UNKNOWN"])
            if previous_query_type == self.QUERY_TYPES["COURSE"]:
                normalized_query, _ = normalize_query(query)
                response = self._generate_response(normalized_query, query_type, requires_history, history_context, context_str)
                answer = response.get("response", "")
                other_info = response.get("other", None)
            elif previous_query_type == self.QUERY_TYPES["PROGRAMME"]:
                prog_name = analysis.get("programme_name", "").strip()
                if prog_name == "":
                    prog_name = self.previous_query_analysis.get("programme_name", "").strip()
                normalized_query, _ = normalize_query(query)
                if master:
                    prog_name = master
                search_results = self.vector_db_client.search_programmes(normalized_query, filter_programme=prog_name if prog_name else None)
                if search_results:
                    context_str, query_information, context_information = self.vector_db_client.build_context_str_programmes(search_results)
                response = self._generate_response(normalized_query, previous_query_type, requires_history, history_context, context_str)
                answer = response.get("response", "")
                other_info = response.get("other", None)
        else:  # unknown query type
            answer = "I'm not sure I fully understood your question. Could you please rephrase it or provide a bit more detail in the next question so I can assist you more effectively? If you are interested in something related to a course, provide the course code; if you are interested in something related to a programme, provide the programme name."
        
        # Update memory with this exchange
        self.query_memory.add(
            user_query=query,
            query_type=query_type,
            additional_info=query_information,
            used_before_info=other_info
        )

        self.context_memory.add(
            user_query=query,
            query_type=query_type,
            retrieved_info=context_information,
            used_before_info=other_info
        )
        
        return answer, {"query_type": query_type, "requires_history": requires_history}
        
    def _generate_response(self, query: str, query_type: str, requires_history: bool, history_context: str, retrieved_info : str) -> str:
        """Generate an LLM-based response when a search is not executed."""
        temperature_set = 0
        if query_type == self.QUERY_TYPES["COURSE"]:
            temperature_set = 0.1
            system_message = "You are an expert academic advisor specialized in university courses."
            prompt = f"""
You are a highly knowledgeable and polite academic assistant with expertise in all DTU courses. Your mission is to provide **accurate**, **detailed**, and **context-aware** answers about DTU courses. Speak in **first person**, as if you are personally assisting the user. Never show internal reasoning or thoughts in the reply — simply respond naturally as a helpful assistant.  

You must always include the **URL of the DTU course page** that was used to generate the answer. The link must be formatted as a markdown link like `[Course Page](https://...)`, and placed **at the end** of your reply inside the `response` field.  
Do **not include just the link** — you must also write the following phrase before it:  
> _"The information has been retrieved from the official course page here: [Course Page](...)"_

At the end, return your reply in **JSON format** with the following structure:
```json
{{
  "response": "Your natural language response in markdown here.",
  "other": "The DTU course code (e.g., '02450') of the course that you used for answering the query, otherwise null. Provide a single course code always."
}}
```

---

### 1. **User Query:**
- This is the most recent request from the user.
- If the query does **not include a course name or course code** or something related to the course, it could be a signal that the answer is in the conversation history.

User Query:
{query}

---

### 2. **Requires Conversation History:**
- This flag indicates whether analyzing the history is required for this query.
- If `True`, look carefully at the conversation history to understand the context, especially if the course is not explicitly mentioned in the current query.

Requires History: {requires_history}

---

### 3. **Course Information:**
- This section contains official and detailed data about DTU courses.
- First, try to match the user query with information in this section, but if the user does not provide any course name or code in the query, try first to figure out if it is referring to the past query, that is in the conversation history.

Course Information:
{retrieved_info}

---

### 4. **Conversation History:**
- This includes recent exchanges between the user and the assistant.
- You should understand from the previous query in the following paragraph if the present query {query} is referring to the past query information.
- If course information is unclear or missing in the query, use this section to infer context from previous user questions or previously retrieved course data.

Conversation History:
{history_context}

---

### ✅ **Your Task:**
1. **Interpret the User Query:**
   - If the course name or code is clearly mentioned, probably you can find the relevant course details from the Course Information section and answer accordingly.
   - If not, do not be creative — first check the conversation history to figure out if the present query {query} is connected to the last query and so you can find information in the conversation history; otherwise, simply ask for more information.
   - For example, if the user first asks "What is the literature of the course 02450" and then "What about the exam?", you can find the information in the conversation history, and so you can answer without asking for more details.

2. **Fallback on History:**
   - If course information is not found in Course Information, check the Conversation History for a reference to the course or topic, mostly in the previous query.

3. **Ask for Clarification (if needed):**
   - If the information cannot be determined from either Course Information or Conversation History, reply in a friendly and polite tone:
     > _"I'm currently unable to identify the course. Could you kindly include the course code when reformulating the question so I can help more effectively?"_

4. **Include the Source URL:**
   - Always include the DTU course page URL in the markdown response.
   - Format it as a markdown link: `[Course Page](https://...)`
   - Do **not** include just the link. You must introduce it with the exact phrase:  
     _"The information has been retrieved from the official course page here: [Course Page](...)"_
   - Always place this phrase + link at the **end of your `response` field**, on a new line if needed.

---

### ✅ **Response Style:**
- Speak **in first person**, as if personally giving suggestions or guidance.
- Always respond in **markdown** format.
- Be **factual**, **concise**, and **professional**.
- Avoid showing internal logic or thought process — just speak naturally and helpfully.
- Return the response in JSON format with the structure provided above.
- The markdown response must:
  - **Always include** the course URL formatted as `[Course Page](https://...)`
  - **Include the following phrase before the link:**  
    _"The information has been retrieved from the official course page here: [Course Page](...)"_
  - **Place the phrase + link at the end of the reply**

---
"""


        elif query_type == self.QUERY_TYPES["PROGRAMME"]:
            temperature_set = 0.3
            system_message = "You are an expert academic advisor specialized in university programmes."
            prompt = f"""
You are a highly knowledgeable and polite academic assistant with expertise in all DTU Master's programmes. Your mission is to provide **accurate**, **detailed**, and **context-aware** answers about DTU Master's programmes. Speak in **first person**, as if you are personally assisting the user. Never show internal reasoning or thoughts in the reply — simply respond naturally as a helpful assistant.

You must always include the **URL of the DTU master's programme page** that was used to generate the answer. The link must be formatted as a markdown link like `[Programme Page](https://...)`, and placed **at the end** of your reply inside the `response` field.  
Do **not include just the link** — you must also write the following phrase before it:  
> _"The information has been retrieved from the official programme page here: [Programme Page](...)"_

At the end, return your reply in **JSON format** with the following structure:
```json
{{
  "response": "Your natural language response in markdown here.",
  "other": "The DTU master's programme name (e.g., 'Business_Analytics') of the master that you used for answering the query, otherwise null. Provide a single master's programme always."
}}
```

---

### 1. **User Query:**
- This is the most recent request from the user.
- If the query does **not include a programme name** or something clearly related to a DTU Master's programme, it may rely on the context provided in the conversation history.

User Query:
{query}

---

### 2. **Requires Conversation History:**
- This flag indicates whether analyzing the history is required for this query.
- If `True`, look carefully at the conversation history to understand the context, especially if the programme name is not explicitly mentioned in the current query.

Requires History: {requires_history}

---

### 3. **Master’s Programme Information:**
- This section contains official and structured information about the DTU Master's programme.
- Try to match the query to one or more fields below. If the programme is not clearly mentioned in the current query, try to infer it from the conversation history.

Master's Programme Data:
{retrieved_info}

---

### 4. **Conversation History:**
- This includes recent exchanges between the user and the assistant.
- Use this to determine whether the current query builds or refers to a previous request about a specific programme.
- If the programme name is unclear or missing in the query, use this section to infer context from previous user questions or previously retrieved master's programme data.

Conversation History:
{history_context}

---

***WARNING***  
Since the user can select a master's programme in the UI sidebar, it can happen that they ask about a different one in the query.  
If {query} refers to a different programme than what is present in the Master's Programme Data, respond with:  
> _"Probably you selected in the sidebar a different master with respect to the one that you are asking for, please check the master's programme name in the sidebar and ask again the question!"_

---

### ✅ **Your Task:**
1. **Interpret the User Query:**
   - If the programme name is clearly mentioned, retrieve the appropriate details from the Master’s Programme Information section.
   - If not, first check the conversation history to determine whether the user is referring to a previously discussed programme.
   - For example, a common situation is when the user asks "What are learning objectives of the master's programme Business Analytics?" and then "What about the curriculum?". In this case, you can find the information in the conversation history and answer without asking for more information.

2. **Identify Relevant Fields:**
   - In curriculum info there are all the informations regarding courses that are mandatory, specializations etc.  
   - ⚠️ **WARNING:** Exams that fall under polytechnical foundation are **mandatory** if asked!

3. **Ask for Clarification (if needed):**
   - If you are unable to identify the programme or the topic, respond politely with:
     > _"I'm currently not sure which Master's programme you're referring to. Could you please include the programme name when reformulating the query so I can help you more effectively?"_

4. **Include the Source URL:**
   - Always include the DTU programme page URL in the markdown response.
   - Format it as a markdown link: `[Programme Page](https://...)`
   - Do **not** include just the link. You must introduce it with the exact phrase:  
     _"The information has been retrieved from the official programme page here: [Programme Page](...)"_
   - Always place this phrase + link at the **end of your `response` field**, on a new line if needed.

---

### ✅ **Response Style:**
- Speak **in first person**, as if personally assisting the user.
- Always respond in **markdown** format.
- Be **factual**, **clear**, and **concise** — but also friendly.
- Never show internal logic or reasoning — only the final helpful response.
- Return the response in JSON format with the structure provided above.
- The markdown response must:
  - **Always include** the programme URL formatted as `[Programme Page](https://...)`
  - **Include the following phrase before the link:**  
    _"The information has been retrieved from the official programme page here: [Programme Page](...)"_
  - **Place the phrase + link at the end of the reply**

---
"""


        else:  # CONVERSATION or UNKNOWN query types
            temperature_set = 0.2
            system_message = "You are an expert academic advisor responding to general queries."
            prompt = f"""
You are a highly knowledgeable and polite academic assistant with expertise in all DTU Master's programmes. Your mission is to provide **accurate**, **detailed**, and **context-aware** answers about DTU Master's programmes. Speak in **first person**, as if you are personally assisting the user. Never show internal reasoning or thoughts in the reply — simply respond naturally as a helpful assistant.

You must always include the **URL of the DTU master's programme page** that was used to generate the answer. The link must be formatted as a markdown link like `[Programme Page](https://...)`, and placed **at the end** of your reply inside the `response` field.  
Do **not include just the link** — you must also write the following phrase before it:  
> _"The information has been retrieved from the official programme page here: [Programme Page](...)"_

At the end, return your reply in **JSON format** with the following structure:
```json
{{
  "response": "Your natural language response in markdown here.",
  "other": "The DTU master's programme name (e.g., 'Business_Analytics') of the master that you used for answering the query, otherwise null. Provide a single master's programme always."
}}
```

---

### 1. **User Query:**
- This is the most recent request from the user.
- If the query does **not include a programme name** or something clearly related to a DTU Master's programme, it may rely on the context provided in the conversation history.

User Query:
{query}

---

### 2. **Requires Conversation History:**
- This flag indicates whether analyzing the history is required for this query.
- If `True`, look carefully at the conversation history to understand the context, especially if the programme name is not explicitly mentioned in the current query.

Requires History: {requires_history}

---

### 3. **Master’s Programme Information:**
- This section contains official and structured information about the DTU Master's programme.
- Try to match the query to one or more fields below. If the programme is not clearly mentioned in the current query, try to infer it from the conversation history.

Master's Programme Data:
{retrieved_info}

---

### 4. **Conversation History:**
- This includes recent exchanges between the user and the assistant.
- Use this to determine whether the current query builds or refers to a previous request about a specific programme.
- If the programme name is unclear or missing in the query, use this section to infer context from previous user questions or previously retrieved master's programme data.

Conversation History:
{history_context}

---

***WARNING***  
Since the user can select a master's programme in the UI sidebar, it can happen that they ask about a different one in the query.  
If {query} refers to a different programme than what is present in the Master's Programme Data, respond with:  
> _"Probably you selected in the sidebar a different master with respect to the one that you are asking for, please check the master's programme name in the sidebar and ask again the question!"_

---

### ✅ **Your Task:**
1. **Interpret the User Query:**
   - If the programme name is clearly mentioned, retrieve the appropriate details from the Master’s Programme Information section.
   - If not, first check the conversation history to determine whether the user is referring to a previously discussed programme.
   - For example, a common situation is when the user asks "What are learning objectives of the master's programme Business Analytics?" and then "What about the curriculum?". In this case, you can find the information in the conversation history and answer without asking for more information.

2. **Identify Relevant Fields:**
   - In curriculum info there are all the informations regarding courses that are mandatory, specializations etc.  
   - ⚠️ **WARNING:** Exams that fall under polytechnical foundation are **mandatory** if asked!

3. **Ask for Clarification (if needed):**
   - If you are unable to identify the programme or the topic, respond politely with:
     > _"I'm currently not sure which Master's programme you're referring to. Could you please include the programme name when reformulating the query so I can help you more effectively?"_

4. **Include the Source URL:**
   - Always include the DTU programme page URL in the markdown response.
   - Format it as a markdown link: `[Programme Page](https://...)`
   - Do **not** include just the link. You must introduce it with the exact phrase:  
     _"The information has been retrieved from the official programme page here: [Programme Page](...)"_
   - Always place this phrase + link at the **end of your `response` field**, on a new line if needed.

---

### ✅ **Response Style:**
- Speak **in first person**, as if personally assisting the user.
- Always respond in **markdown** format.
- Be **factual**, **clear**, and **concise** — but also friendly.
- Never show internal logic or reasoning — only the final helpful response.
- Return the response in JSON format with the structure provided above.
- The markdown response must:
  - **Always include** the programme URL formatted as `[Programme Page](https://...)`
  - **Include the following phrase before the link:**  
    _"The information has been retrieved from the official programme page here: [Programme Page](...)"_
  - **Place the phrase + link at the end of the reply**

---
"""


        try:
            response = self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature_set,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content

            # Additional safety check
            if not content.strip().startswith("{"):
                raise ValueError("Response is not valid JSON")
            json_answer = json.loads(content)
            return json_answer
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error while processing your question. Could you try asking in a different way?"

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Get embedding vector for text using OpenAI's API."""
    # Add robust null checking
    if text is None:
        print("Warning: Received None input in get_embedding, using empty string")
        text = ""
    elif not isinstance(text, str):
        print(f"Warning: Expected string in get_embedding, got {type(text)}")
        text = str(text) if text is not None else ""
        
    if not text.strip():
        return [0.0] * 1536
    try:
        response = openai_client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0.0] * 1536

def get_name_embedding(text: str) -> List[float]:
    """Get a name-specific embedding using SentenceTransformer."""
    # Add robust null checking
    if text is None:
        print("Warning: Received None input in get_name_embedding, using empty string")
        text = ""
    elif not isinstance(text, str):
        print(f"Warning: Expected string in get_name_embedding, got {type(text)}")
        text = str(text) if text is not None else ""
        
    try:
        embedding = name_embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating name embedding: {e}")
        return [0.0] * 384

def display_markdown(text: str) -> None:
    """Display text as markdown in the console."""
    console.print(Markdown(text))

def interactive_chat():
    """Run an interactive chat session with the query router and response generation."""
    memory_context = Memory(max_items=10)
    memory_query = QueryMemory(max_items=10)
    weaviate_client = MultiVectorDBClient(vector_db_configs)
    router = QueryRouter(memory_query, memory_context, weaviate_client)
    
    print("\n🎓 Welcome to the Academic Assistant!")
    print("Ask me questions about courses or programmes. Type 'exit' to quit or 'help' for more options.\n")
    display_markdown("## Commands:\n* `exit`: Quit the chat\n* `help`: Show help\n* `clear`: Clear conversation history")
    
    try:
        while True:
            user_query = input("\n👤 User: ")
            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
            elif user_query.lower() == 'help':
                display_markdown("## Commands:\n* `exit`: Quit the chat\n* `help`: Show help\n* `clear`: Clear conversation history")
                continue
            elif user_query.lower() == 'clear':
                memory_context.clear()
                memory_query.clear()
                print("🧹 Conversation history cleared.")
                continue
            
            print("\n🤖 Assistant: ", end="")
            response, analysis = router.route_query(user_query)
            display_markdown(response)
    except KeyboardInterrupt:
        print("\n\n👋 Chat session interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    try:
        if not OPENAI_API_KEY:
            console.print("[bold red]Error:[/bold red] OPENAI_API_KEY environment variable is not set.")
            console.print("Please set it using: export OPENAI_API_KEY='your-api-key'")
            exit(1)
        interactive_chat()
    except KeyboardInterrupt:
        console.print("\n\n[bold]Program interrupted. Exiting...[/bold]")
    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {e}")
        import traceback
        traceback.print_exc()