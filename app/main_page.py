import sys
import os
import traceback
import tracemalloc
import streamlit as st
from streamlit_star_rating import st_star_rating
from PIL import Image
import base64
import datetime
import uuid
import pymongo
from pymongo import MongoClient

os.environ["STREAMLIT_WATCH_MODE"] = "poll"
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNING"] = "true"
tracemalloc.start()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from rag_dtu.routing.query_router import Memory, QueryMemory, MultiVectorDBClient, QueryRouter
except ImportError:
    print("Error importing modules. Please ensure the 'rag_dtu' package is installed and accessible.")

# MongoDB Connection Setup
DB_PASSWORD = st.secrets["DB_PASSWORD"]  # Replace this with your actual password
DB_USERNAME = st.secrets["DB_USERNAME"]

CONNECTION_STRING = f"mongodb+srv://{DB_USERNAME}:{DB_PASSWORD}@userfeedback.iurtfej.mongodb.net/?retryWrites=true&w=majority&appName=UserFeedback"
# Function to get MongoDB connection
def get_mongodb_connection():
    try:
        client = MongoClient(CONNECTION_STRING)
        # Test connection by accessing server info
        client.server_info()
        return client
    except pymongo.errors.ServerSelectionTimeoutError as e:
        st.error(f"Could not connect to MongoDB: {e}")
        return None
    except pymongo.errors.OperationFailure as e:
        st.error(f"Authentication failed: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error connecting to MongoDB: {e}")
        return None

# Define program names mapping - displayed name to internal format
PROGRAM_MAPPING = {
    "Select a program": None,
    "MSc Engineering Light": "Engineering_Light",
    "MSc Biomaterial Engineering for Medicine": "Biomaterial_Engineering_for_Medicine",
    "MSc Chemical and Biochemical Engineering": "Chemical_and_Biochemical_Engineering",
    "MSc Earth and Space Physics and Engineering": "Earth_and_Space_Physics_and_Engineering",
    "MSc Pharmaceutical Design and Engineering": "Pharmaceutical_Design_and_Engineering",
    "MSc Transport and Logistics": "Transport_and_Logistics",
    "MSc Design and Innovation": "Design_and_Innovation",
    "MSc Food Technology": "Food_Technology",
    "MSc Communication Technologies and System Design": "Communication_Technologies_and_System_Design",
    "MSc Applied Chemistry": "Applied_Chemistry",
    "MSc Sustainable Energy": "Sustainable_Energy",
    "MSc Quantitative Biology and Disease Modelling": "Quantitative_Biology_and_Disease_Modelling",
    "MSc Biotechnology": "Biotechnology",
    "MSc Industrial Engineering and Management": "Industrial_Engineering_and_Management",
    "MSc Wind Energy": "Wind_Energy",
    "MSc Engineering Acoustics": "Engineering_Acoustics",
    "MSc Sustainable Fisheries and Aquaculture": "Sustainable_Fisheries_and_Aquaculture",
    "MSc Mathematical Modelling and Computation": "Mathematical_Modelling_and_Computation",
    "MSc Engineering Physics": "Engineering_Physics",
    "MSc Biomedical Engineering": "Biomedical_Engineering",
    "MSc Technology Entrepreneurship": "Technology_Entrepreneurship",
    "MSc Materials and Manufacturing Engineering": "Materials_and_Manufacturing_Engineering",
    "MSc Autonomous Systems": "Autonomous_Systems",
    "MSc Civil Engineering": "Civil_Engineering",
    "MSc Electrical Engineering": "Electrical_Engineering",
    "MSc Human-oriented Artificial Intelligence": "Human-oriented_Artificial_Intelligence",
    "MSc Bioinformatics and Systems": "Bioinformatics_and_Systems",
    "MSc Mechanical Engineering": "Mechanical_Engineering",
    "MSc Petroleum Engineering": "Petroleum_Engineering",
    "MSc Computer Science and Engineering": "Computer_Science_and_Engineering",
    "MSc Business Analytics": "Business_Analytics",
    "MSc Architectural Engineering": "Architectural_Engineering",
}

# Get list of display names for the dropdown
MASTERS_PROGRAMS = list(PROGRAM_MAPPING.keys())

if "router" not in st.session_state:
    vector_db_configs = {
        "programme_db": {
            "url": "https://yz34awbrqlko1tvblm77g.c0.europe-west3.gcp.weaviate.cloud",
            "api_key": "A6IFGm879tMitR94FWoffvNsBDeMTu8eZv8n",
            "collection_name": "Chunk"
        },
        "course_db": {
            "url": "https://yz34awbrqlko1tvblm77g.c0.europe-west3.gcp.weaviate.cloud",
            "api_key": "YVldI0WBz6MUZVoPZA5wp5t7zalMI12jdkfm",
            "collection_name": "Course"
        }
    }

    memory_context = Memory(max_items=10)
    memory_query = QueryMemory(max_items=10)
    weaviate_client = MultiVectorDBClient(vector_db_configs)
    st.session_state.router = QueryRouter(memory_query, memory_context, weaviate_client)
    st.session_state.memory_context = memory_context
    st.session_state.memory_query = memory_query

router = st.session_state.router
memory_context = st.session_state.memory_context
memory_query = st.session_state.memory_query

welcome_message = """
#### **Welcome to the DTU Chat Assistant!**

👋 Hi there, and welcome to your personal **DTU Chat Assistant** — built to make your academic life at the Technical University of Denmark easier and more intuitive.

Whether you're starting your Master's or in the middle of your studies, this assistant is here to guide you through:

- 🔹 **Master's program structures** — Learn about your specific program's mandatory courses, elective options, and specialization tracks.  
- 🔹 **Course information** — Get quick details on ECTS credits, prerequisites, instructors, or semester placement.  
- 🔹 **Exam requirements** — Know which courses are mandatory for graduation and when the exams typically take place.  
- 🔹 **Cross-program exploration** — Curious about courses from other DTU programs? I can help you see how they align with your study plan.  
- 🔹 **Recommendations & eligibility** — Wondering if you meet the prerequisites for a course? Just ask.

💡 You can also select your Master's program from the sidebar to get tailored answers just for you.

If you're not sure how to start, try asking:
- *"What are the mandatory courses for MSc Computer Science?"*  
- *"Can I take 01005 Linear Algebra as an elective?"*  
- *"How many ECTS do I need to graduate?"*

**Ready when you are — just type your question below and let's get started! 🎓**
"""

st.set_page_config(page_title="DTU CHAT", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to improve rating display
st.markdown("""
<style>
html[data-theme="light"] .fixed-header {
    background-color: white;
    border-bottom: 1px solid #ddd;
}
html[data-theme="dark"] .fixed-header {
    background-color: #0e1117;
    border-bottom: 1px solid #333;
}
.fixed-header {
    position: fixed;
    top: 100px;
    left: 0;
    right: 0;
    width: 100%;
    z-index: 1000;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
}
.fixed-header h1 {
    font-size: 2.2rem;
    margin: 0;
    color: inherit;
    text-align: center;
}
.page-content {
    margin-top: 100px;
}
.chat-container {
    border-radius: 10px;
    padding: 20px;
    background-color: #f9f9f9;
    min-height: 400px;
    max-height: 65vh;
    overflow-y: auto;
    margin-bottom: 80px;
}
.stButton button {
    background-color: #990000;
    color: white;
}
.program-badge {
    background-color: #990000;
    color: white;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 0.9rem;
    display: inline-block;
    margin-bottom: 15px;
}

/* Completely redesigned rating section styling */
.rating-section {
    margin-top: 15px;
    border-radius: 8px;
    border: 1px solid;
    overflow: hidden;
}

/* Light theme styling */
html[data-theme="light"] .rating-section {
    background-color: rgba(248, 249, 250, 0.7);
    border-color: #ddd;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

/* Dark theme styling */
html[data-theme="dark"] .rating-section {
    background-color: rgba(30, 33, 48, 0.7);
    border-color: #444;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}

/* Rating header styling */
.rating-header {
    padding: 8px 15px;
    border-bottom: 1px solid;
    display: flex;
    align-items: center;
}

html[data-theme="light"] .rating-header {
    background-color: #f2f2f2;
    border-color: #ddd;
}

html[data-theme="dark"] .rating-header {
    background-color: #2c2c3a;
    border-color: #444;
}

.rating-title {
    font-weight: 600;
    font-size: 14px;
    margin: 0;
    display: flex;
    align-items: center;
}

.rating-title svg {
    margin-right: 6px;
}

/* Rating content styling */
.rating-content {
    padding: 12px 15px;
}

/* Style the stars to be more visible */
.streamlit-star-rating {
    margin-bottom: 12px !important;
}

/* Style text area */
.stTextArea textarea {
    border-radius: 6px;
    resize: vertical;
}

html[data-theme="light"] .stTextArea textarea {
    border-color: #ddd;
    background-color: white;
}

html[data-theme="dark"] .stTextArea textarea {
    border-color: #555;
    background-color: #1e1e2e;
}

/* Submit button styling */
.rating-submit button {
    background-color: #990000 !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.rating-submit button:hover:not(:disabled) {
    background-color: #b30000 !important;
    box-shadow: 0 2px 5px rgba(153, 0, 0, 0.3) !important;
}

.rating-submit button:disabled {
    background-color: #cccccc !important;
    color: #888888 !important;
    cursor: not-allowed !important;
}

/* Success message styling */
.rating-success {
    background-color: #d4edda;
    color: #155724;
    padding: 8px 12px;
    border-radius: 4px;
    margin-top: 8px;
    font-size: 14px;
    display: flex;
    align-items: center;
}

.rating-success svg {
    margin-right: 6px;
}

/* Remove default Streamlit container padding/margins within rating section */
.rating-section .stTextArea label {
    font-size: 14px;
    font-weight: normal;
    margin-bottom: 5px;
}

/* Make success message more visible on dark theme */
html[data-theme="dark"] .rating-success {
    background-color: rgba(40, 167, 69, 0.2);
    color: #75b798;
}

/* Fix background issue with streamlit chat container */
.element-container {
    background: transparent !important;
}

.stChatMessageContent {
    background: transparent !important;
}

/* Support for chat messages transparency - making rating form stand out */
.stChatMessage .stChatMessageContent {
    background-color: transparent !important;
}

/* Remove any default Streamlit card backgrounds */
div[data-testid="stChatMessageContent"] > div {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-content">', unsafe_allow_html=True)

def clear_chat():
    st.session_state.messages = []
    memory_context.clear()
    memory_query.clear()
    st.session_state.active_program = None
    st.session_state.active_program_internal = None
    st.session_state.selected_program = "Select a program"
    # Add the welcome message back after clearing
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

def exit_chat():
    st.session_state.messages = []
    memory_context.clear()
    memory_query.clear()
    st.session_state.chat_active = False

def save_rating(message_idx, rating, comment=""):
    # Update the rating in the session state
    st.session_state.messages[message_idx]["rating"] = rating
    st.session_state.messages[message_idx]["comment"] = comment
    
    # Get the associated query and response
    if message_idx > 0:
        # Find the preceding user message
        query_idx = message_idx - 1
        if query_idx >= 0 and st.session_state.messages[query_idx]["role"] == "user":
            query = st.session_state.messages[query_idx]["content"]
        else:
            query = "N/A"  # No matching query found
    else:
        query = "N/A"  # Welcome message
    
    response = st.session_state.messages[message_idx]["content"]
    
    # Generate a unique ID for this rating
    rating_id = str(uuid.uuid4())
    
    # Get current session ID or create one
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Generate a unique message ID if not exists
    if "message_id" not in st.session_state.messages[message_idx]:
        st.session_state.messages[message_idx]["message_id"] = str(uuid.uuid4())
    
    message_id = st.session_state.messages[message_idx]["message_id"]
    
    # Get program selection if available
    program = st.session_state.active_program if st.session_state.active_program else "No program selected"
    
    # Get current timestamp
    timestamp = datetime.datetime.now()
    
    # Create rating document
    rating_doc = {
        "rating_id": rating_id,
        "session_id": st.session_state.session_id,
        "message_id": message_id,
        "query": query,
        "response": response,
        "rating": rating,
        "comment": comment,
        "program": program,
        "timestamp": timestamp
    }
    
    # Save to MongoDB
    try:
        # Get MongoDB connection
        mongo_client = get_mongodb_connection()
        if mongo_client:
            db = mongo_client.get_database("dtu_feedback")
            ratings_collection = db.get_collection("chat_ratings")
            
            # Insert the rating document
            ratings_collection.insert_one(rating_doc)
            
            # Close connection
            mongo_client.close()
            return True
        else:
            st.error("Failed to connect to MongoDB. Rating not saved.")
            return False
    except Exception as e:
        st.error(f"Failed to save rating: {e}")
        return False

def update_master_program():
    selected = st.session_state.selected_program
    if selected != "Select a program":
        st.session_state.active_program = selected
        # Store the internal representation for query routing
        st.session_state.active_program_internal = PROGRAM_MAPPING[selected]
        st.success(f"Your program is set to: {selected}")
    else:
        st.session_state.active_program = None
        st.session_state.active_program_internal = None

def deselect_program():
    st.session_state.active_program = None
    st.session_state.active_program_internal = None
    st.session_state.selected_program = "Select a program"

# Initialize the messages list only if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# Create a session ID if not exists
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "active_program" not in st.session_state:
    st.session_state.active_program = None
if "active_program_internal" not in st.session_state:
    st.session_state.active_program_internal = None
if "selected_program" not in st.session_state:
    st.session_state.selected_program = "Select a program"
if "pending_ratings" not in st.session_state:
    st.session_state.pending_ratings = {}
if "rating_success" not in st.session_state:
    st.session_state.rating_success = {}

# Test MongoDB connection at startup
if "mongo_connected" not in st.session_state:
    client = get_mongodb_connection()
    if client:
        st.session_state.mongo_connected = True
        client.close()
    else:
        st.session_state.mongo_connected = False

with st.sidebar:
    st.markdown('<h1 style="font-size: 2.5rem;"><span style="color:#990000;">DTU</span> Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### 🤖 About This Chatbot")
    st.markdown("This chatbot is designed to support DTU Master's students by making it easier to access and understand key academic information.")
    st.markdown("We know that navigating a study program can be confusing. That's why we built this assistant to help answer questions like: What are the prerequisites for a specific course? Is a certain course mandatory for my Master's program?")
    st.markdown("Whether you're figuring out program requirements, mandatory exams, or course details like ECTS, prerequisites, or timetables—this assistant is here to help. Ask away!")

    # Show database connection status in sidebar
    if st.session_state.mongo_connected:
        st.success("✅ MongoDB Connected")
    else:
        st.error("❌ MongoDB Connection Failed")

    st.markdown("### 🎓 Your Program")
    st.markdown("Select your Master's program")
    st.selectbox("Select your Master's program", MASTERS_PROGRAMS, key="selected_program", on_change=update_master_program, label_visibility="collapsed")
    if st.session_state.active_program:
        st.info(f"Current program: {st.session_state.active_program}")
        st.button("🔄 Change Program", on_click=deselect_program)

    st.markdown("### 🔄 Chat Controls")
    st.button("Clear Chat", on_click=clear_chat, use_container_width=True)
        
# Display chat messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Only show rating widget for assistant messages (not the welcome message)
        if message["role"] == "assistant" and idx > 0:
            # Create unique keys for this message's widgets
            rating_key = f"rating_{idx}_{id(message)}"
            comment_key = f"comment_{idx}_{id(message)}"
            submit_key = f"submit_{idx}_{id(message)}"
            success_key = f"success_{idx}_{id(message)}"
            
            # Check if we should show a success message
            show_success = success_key in st.session_state.rating_success and st.session_state.rating_success[success_key]
            
            # Get current rating if exists
            current_rating = message.get("rating", 0)
            current_comment = message.get("comment", "")
            
            # Use a dedicated container for ratings to style it better
            rating_container = st.container()
            
            with rating_container:
                st.markdown('<div class="rating-section">', unsafe_allow_html=True)
                
                # Rating header
                st.markdown("""
                <div class="rating-header">
                    <p class="rating-title">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 2L15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2z"></path>
                        </svg>
                        Rate this response
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Rating content
                st.markdown('<div class="rating-content">', unsafe_allow_html=True)
                
                # Star rating widget
                rating = st_star_rating("", maxValue=5, defaultValue=current_rating, key=rating_key, size=24)
                
                # Comment text area
                comment = st.text_area("Add a comment (optional):", value=current_comment, key=comment_key, height=75)
                
                # Submit button - only active if rating has changed or is greater than 0
                st.markdown('<div class="rating-submit">', unsafe_allow_html=True)
                if st.button("Submit Feedback", key=submit_key, disabled=(rating == 0)):
                    if save_rating(idx, rating, comment):
                        st.session_state.rating_success[success_key] = True
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Success message if rating was just submitted
                if show_success:
                    st.markdown("""
                    <div class="rating-success">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        Thank you for your feedback!
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Message input
prompt = st.chat_input("What can I help with?")
if prompt:
    # Generate a unique ID for this message
    message_id = str(uuid.uuid4())
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "message_id": message_id})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.active_program_internal:
                    response, _ = router.route_query(prompt, st.session_state.active_program_internal)
                else:
                    response, _ = router.route_query(prompt)
                st.markdown(response)
                
                # Generate a unique ID for the response
                response_id = str(uuid.uuid4())
                st.session_state.messages.append({"role": "assistant", "content": response, "message_id": response_id})
                
                # Force a rerun to show the rating widget immediately
                st.rerun()
                
            except Exception as e:
                response = f"❌ Error: {e}"
                traceback.print_exc()
                st.error(response)
                st.session_state.messages.append({"role": "assistant", "content": response, "message_id": str(uuid.uuid4())})

st.markdown("""
<script>
const chatContainer = document.getElementById("chat-container");
if (chatContainer) {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}
</script>
</div>
""", unsafe_allow_html=True)