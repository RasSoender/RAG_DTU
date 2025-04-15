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
    }
    .sidebar-content {
        padding: 20px 10px;
    }
    .stButton button {
        background-color: #990000;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Functions for session state management
def clear_chat():
    st.session_state.messages = []
    memory_context.clear()
    memory_query.clear()

def exit_chat():
    st.session_state.messages = []
    memory_context.clear()
    memory_query.clear()
    st.session_state.chat_active = False

def start_new_chat():
    st.session_state.chat_active = True
    st.session_state.messages = []

def save_rating(rating):
    st.session_state.last_rating = rating
    st.success(f"Thanks for your rating: {rating} stars!")

def update_master_program():
    selected = st.session_state.selected_program
    if selected != "Select a program":
        st.session_state.active_program = selected
        st.success(f"Your program is set to: {selected}")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_active" not in st.session_state:
    st.session_state.chat_active = True
    
if "active_program" not in st.session_state:
    st.session_state.active_program = None
    
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
    
    st.markdown("### 🔄 Chat Controls")
    col1, col2 = st.columns(2)
    with col1:
        clear_button = st.button("🧹 Clear Chat", on_click=clear_chat)
    with col2:
        exit_button = st.button("🚪 Exit Chat", on_click=exit_chat)

# Main content
if not st.session_state.chat_active:
    st.markdown('<h1 class="main-header">Welcome to DTU Chat Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Your guide to navigating DTU programs and courses</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://www.dtu.dk/-/media/DTUdk/DTU-generelt/CampusLife/DTUcampuslifestudents-Joachim-Rode.jpg", use_column_width=True)
        st.button("Start a new conversation", on_click=start_new_chat, key="start_button")
    
else:
    st.markdown('<h1 class="main-header"><span style="color:#990000;">DTU</span> Chatbot</h1>', unsafe_allow_html=True)
    
    if st.session_state.active_program:
        st.markdown(f"<p>Currently assisting with: <strong>{st.session_state.active_program}</strong></p>", unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Add rating option for assistant messages
                if message["role"] == "assistant" and "rating" not in message:
                    message_idx = st.session_state.messages.index(message)
                    with st.expander("Rate this response"):
                        rating = st_star_rating(
                            label="How helpful was this response?", 
                            maxValue=5,
                            defaultValue=0,
                            key=f"rating_{message_idx}"
                        )
                        if rating > 0:
                            st.button(
                                "Submit Rating", 
                                key=f"submit_rating_{message_idx}",
                                on_click=save_rating,
                                args=(rating,)
                            )
                            message["rating"] = rating
    
    # Chat input
    if prompt := st.chat_input("What can I help with?"):
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

# Footer
footer = """
<div class="footer">
    <div style="float: left;">Need help? Contact: <a href="mailto:support@dtu.dk">support@dtu.dk</a></div>
    <div style="float: right;">DTU Student Services: +45 45 25 11 75</div>
    <div style="clear: both;"></div>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)