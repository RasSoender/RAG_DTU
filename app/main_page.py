import sys
import os
import streamlit as st
from streamlit_star_rating import st_star_rating
from PIL import Image
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from rag_dtu.routing.query_router import Memory, QueryMemory, MultiVectorDBClient, QueryRouter
except ImportError:
    # Mock classes for development purposes
    class Memory:
        def __init__(self, max_items=10):
            self.max_items = max_items
            self.items = []
        def clear(self):
            self.items = []
            
    class QueryMemory(Memory):
        pass
        
    class MultiVectorDBClient:
        def __init__(self, configs):
            self.configs = configs
            
    class QueryRouter:
        def __init__(self, memory_query, memory_context, client):
            self.memory_query = memory_query
            self.memory_context = memory_context
            self.client = client
            
        def route_query(self, query):
            # Mock response
            return f"This is a response to: {query}", {}

# Vector DB configurations
vector_db_configs = {
    "programme_db": {
        "url": "https://fjax7aot34bgxxo433ma.c0.europe-west3.gcp.weaviate.cloud",
        "api_key": "4JjcaYcEYBUzb46TpPu6f1qR5CjVJXb5wFB7",
        "collection_name": "Chunk"  # Programme collection
    },
    "course_db": {
        "url": "https://yz34awbrqlko1tvblm77g.c0.europe-west3.gcp.weaviate.cloud",
        "api_key": "YVldI0WBz6MUZVoPZA5wp5t7zalMI12jdkfm",
        "collection_name": "Course"
    }
}

# Initialize components
memory_context = Memory(max_items=10)
memory_query = QueryMemory(max_items=10)
weaviate_client = MultiVectorDBClient(vector_db_configs)
router = QueryRouter(memory_query, memory_context, weaviate_client)

# Define available masters programs
MASTERS_PROGRAMS = [
    "Select a program",
    "MSc Computer Science",
    "MSc Electrical Engineering",
    "MSc Sustainable Energy",
    "MSc Biotechnology",
    "MSc Mathematical Modelling",
    "MSc Digital Media Engineering",
    "MSc Engineering Management"
]

# Styling and layout
st.set_page_config(
    page_title="DTU CHAT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #990000;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .chat-container {
        border-radius: 10px;
        padding: 20px;
        background-color: #f9f9f9;
        margin-bottom: 80px; /* Increased space for the footer and input bar */
        min-height: 400px; /* Ensure minimum height even when empty */
    }
    .feedback-container {
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 5px;
        margin-top: 10px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f0f0f0;
        padding: 10px 20px;
        text-align: center;
        border-top: 1px solid #ddd;
        z-index: 900;
    }
    .input-area {
        position: fixed;
        bottom: 50px; /* Position above footer */
        left: 0;
        width: 100%;
        padding: 10px;
        background-color: white;
        z-index: 950;
        border-top: 1px solid #eee;
    }
    .sidebar-content {
        padding: 20px 10px;
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
    .main-content {
        padding-bottom: 140px; /* Extra space for input and footer */
    }
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 70vh;
        text-align: center;
    }
    /* Make the rating stars bigger */
    .stStarRating svg {
        height: 30px !important;
        width: 30px !important;
    }
    /* Space for the chat input that's always visible */
    .block-container {
        padding-bottom: 120px;
    }
    /* Hide the default streamlit chat input when we use a custom one */
    [data-testid="stChatInput"] {
        position: fixed;
        bottom: 60px;
        width: calc(100% - 80px); /* Account for sidebar width */
        margin-left: 15px;
        margin-right: 15px;
        z-index: 999;
        background-color: white;
        border-radius: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    /* Adjust rating expander to be more visible */
    .st-expander {
        background-color: #f8f8f8;
        border-radius: 8px;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Functions for session state management
def clear_chat():
    st.session_state.messages = []
    memory_context.clear()
    memory_query.clear()
    # Reset program selection
    st.session_state.active_program = None
    st.session_state.selected_program = "Select a program"

def exit_chat():
    st.session_state.messages = []
    memory_context.clear()
    memory_query.clear()
    st.session_state.chat_active = False

def start_new_chat():
    st.session_state.chat_active = True
    st.session_state.messages = []

def save_rating(message_idx, rating):
    # Update the rating in the specific message
    st.session_state.messages[message_idx]["rating"] = rating
    st.success(f"Thanks for your rating: {rating} stars!")

def update_master_program():
    selected = st.session_state.selected_program
    if selected != "Select a program":
        st.session_state.active_program = selected
        st.success(f"Your program is set to: {selected}")
    else:
        st.session_state.active_program = None

def deselect_program():
    st.session_state.active_program = None
    st.session_state.selected_program = "Select a program"

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_active" not in st.session_state:
    st.session_state.chat_active = True
    
if "active_program" not in st.session_state:
    st.session_state.active_program = None
    
if "selected_program" not in st.session_state:
    st.session_state.selected_program = "Select a program"
    
if "last_rating" not in st.session_state:
    st.session_state.last_rating = None

# Sidebar content
with st.sidebar:
    st.image("https://www.dtu.dk/-/media/DTUdk/DTU-generelt/Grafisk-identitet/dtulogo_rect_red_digital.svg", width=200)
    
    st.markdown("### 🤖 About This Chatbot")
    st.markdown(
        "This chatbot is designed to support DTU Master's students by making it "
        "easier to access and understand key academic information."
    )
    
    st.markdown(
        "We know that navigating a study program can be confusing. That's why we built "
        "this assistant to help answer questions like: What are the prerequisites for a "
        "specific course? Is a certain course mandatory for my Master's program?"
    )
    
    st.markdown(
        "Whether you're figuring out program requirements, mandatory exams, or course details "
        "like ECTS, prerequisites, or timetables—this assistant is here to help. Ask away!"
    )
    
    st.markdown("### 🎓 Your Program")
    selected_program = st.selectbox(
        "Select your Master's program:",
        MASTERS_PROGRAMS,
        key="selected_program",
        on_change=update_master_program
    )
    
    if st.session_state.active_program:
        st.info(f"Current program: {st.session_state.active_program}")
        st.button("🔄 Change Program", on_click=deselect_program)
    
    st.markdown("### 🔄 Chat Controls")
    col1, col2 = st.columns(2)
    with col1:
        clear_button = st.button("🧹 Clear Chat", on_click=clear_chat)
    with col2:
        exit_button = st.button("🚪 Exit Chat", on_click=exit_chat)

# Main content with proper structure
main_content = st.container()
with main_content:
    if not st.session_state.chat_active:
        st.markdown('<div class="welcome-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="main-header">Welcome to DTU Chat Assistant</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subheader">Your guide to navigating DTU programs and courses</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://www.dtu.dk/-/media/DTUdk/DTU-generelt/CampusLife/DTUcampuslifestudents-Joachim-Rode.jpg", use_column_width=True)
            st.button("Start a new conversation", on_click=start_new_chat, key="start_button")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.markdown('<h1 class="main-header"><span style="color:#990000;">DTU</span> Chatbot</h1>', unsafe_allow_html=True)
        
        if st.session_state.active_program:
            st.markdown(f'<div class="program-badge">📚 {st.session_state.active_program}</div>', unsafe_allow_html=True)
        
        # Chat container
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for idx, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Add rating option for assistant messages with unique keys
                    if message["role"] == "assistant":
                        # Create a unique key for each message's rating system
                        rating_key = f"rating_{idx}_{id(message)}"
                        button_key = f"button_{idx}_{id(message)}"
                        
                        with st.expander("📊 Rate this response"):
                            current_rating = message.get("rating", 0)
                            rating = st_star_rating(
                                label="How helpful was this response?", 
                                maxValue=5,
                                defaultValue=current_rating,
                                key=rating_key
                            )
                            if rating > 0 and rating != current_rating:
                                st.button(
                                    "Submit Rating", 
                                    key=button_key,
                                    on_click=save_rating,
                                    args=(idx, rating)
                                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input - always visible at bottom
        prompt = st.chat_input("What can I help with?")
        if prompt:
            # User message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
        
            # Bot message
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Add program context if available
                        if st.session_state.active_program and st.session_state.active_program != "Select a program":
                            context_prompt = f"[User's program: {st.session_state.active_program}] {prompt}"
                            response, _ = router.route_query(context_prompt)
                        else:
                            response, _ = router.route_query(prompt)
                        st.markdown(response)
                    except Exception as e:
                        response = f"❌ Error: {e}"
                        st.error(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force a rerun to update the UI immediately
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
footer = """
<div class="footer">
    <div style="float: left;">Need help? Contact: <a href="mailto:support@dtu.dk">support@dtu.dk</a></div>
    <div style="float: right;">DTU Student Services: +45 45 25 11 75</div>
    <div style="clear: both;"></div>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)