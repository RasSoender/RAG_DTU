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
            return f"This is a response to: {query}", {}

vector_db_configs = {
    "programme_db": {
        "url": "https://fjax7aot34bgxxo433ma.c0.europe-west3.gcp.weaviate.cloud",
        "api_key": "4JjcaYcEYBUzb46TpPu6f1qR5CjVJXb5wFB7",
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
router = QueryRouter(memory_query, memory_context, weaviate_client)

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

st.set_page_config(page_title="DTU CHAT", layout="wide", initial_sidebar_state="expanded")

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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-content">', unsafe_allow_html=True)

def clear_chat():
    st.session_state.messages = []
    memory_context.clear()
    memory_query.clear()
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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False
if "active_program" not in st.session_state:
    st.session_state.active_program = None
if "selected_program" not in st.session_state:
    st.session_state.selected_program = "Select a program"
if "last_rating" not in st.session_state:
    st.session_state.last_rating = None

with st.sidebar:
    st.markdown('<h1 style="font-size: 2.5rem;"><span style="color:#990000;">DTU</span> Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### 🤖 About This Chatbot")
    st.markdown("This chatbot is designed to support DTU Master's students by making it easier to access and understand key academic information.")
    st.markdown("We know that navigating a study program can be confusing. That's why we built this assistant to help answer questions like: What are the prerequisites for a specific course? Is a certain course mandatory for my Master's program?")
    st.markdown("Whether you're figuring out program requirements, mandatory exams, or course details like ECTS, prerequisites, or timetables—this assistant is here to help. Ask away!")

    st.markdown("### 🎓 Your Program")
    st.markdown("Select your Master's program")
    st.selectbox("Select your Master's program", MASTERS_PROGRAMS, key="selected_program", on_change=update_master_program, label_visibility="collapsed")
    if st.session_state.active_program:
        st.info(f"Current program: {st.session_state.active_program}")
        st.button("🔄 Change Program", on_click=deselect_program)

    st.markdown("### 🔄 Chat Controls")
    col1, col2 = st.columns(2)
    with col1:
        st.button("🧹 Clear Chat", on_click=clear_chat)
    with col2:
        st.button("🚪 Exit Chat", on_click=exit_chat)

if not st.session_state.chat_active:
    st.markdown('<div style="text-align: center; margin-top: 5rem;">', unsafe_allow_html=True)
    st.subheader("Your guide to navigating DTU programs and courses")
    st.button("Start a new conversation", on_click=start_new_chat, key="start_button")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # st.markdown('<div id="chat-container" class="chat-container">', unsafe_allow_html=True)

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                rating_key = f"rating_{idx}_{id(message)}"
                button_key = f"button_{idx}_{id(message)}"
                with st.expander("📊 Rate this response"):
                    current_rating = message.get("rating", 0)
                    rating = st_star_rating("How helpful was this response?", maxValue=5, defaultValue=current_rating, key=rating_key, size=20, customCSS = "h3{font-size: 15px;}")
                    if rating > 0 and rating != current_rating:
                        st.button("Submit Rating", key=button_key, on_click=save_rating, args=(idx, rating))
    # st.markdown('</div>', unsafe_allow_html=True)

    prompt = st.chat_input("What can I help with?")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
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
        st.rerun()

st.markdown("""
<script>
const chatContainer = document.getElementById("chat-container");
if (chatContainer) {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}
</script>
</div>
""", unsafe_allow_html=True)
