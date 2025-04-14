import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import streamlit as st
from rag_dtu.routing.query_router import Memory, QueryMemory, MultiVectorDBClient, QueryRouter


vector_db_configs = {
    "programme_db": {
        "url": "https://fjax7aot34bgxxo433ma.c0.europe-west3.gcp.weaviate.cloud",
        "api_key": "4JjcaYcEYBUzb46TpPu6f1qR5CjVJXb5wFB7",
        "collection_name": "Chunk"  # Programme collection
    },
    "course_db": {
        "url": "https://yz34awbrqlko1tvblm77g.c0.europe-west3.gcp.weaviate.cloud",
        "api_key": "YVldI0WBz6MUZVoPZA5wp5t7zalMI12jdkfm",  # Replace with your second API key
        "collection_name": "Course"
    }
}

memory_context = Memory(max_items=10)
memory_query = QueryMemory(max_items=10)
weaviate_client = MultiVectorDBClient(vector_db_configs)
router = QueryRouter(memory_query, memory_context, weaviate_client)


# Page config
st.set_page_config(page_title="DTU CHAT", layout="wide")


def clear_chat():
    st.session_state.messages = []


with st.sidebar:
    st.markdown("🤖 About This Chatbot") 
    st.markdown("This chatbot is designed to support DTU Master's students by making it " \
    "easier to access and understand key academic information.")
    st.markdown("We know that navigating a study program can be confusing. That’s why we built" \
                "this assistant to help answer questions like: What are the prerequisites for a " \
                "specific course? Is a certain course mandatory for my Master’s program?")
    st.markdown("Whether you're figuring out program requirements, mandatory exams, or course details " \
                "like ECTS, prerequisites, or timetables—this assistant is here to help. Ask away!")
    clear_button = st.button("Clear chat", on_click=clear_chat)

st.title(":red[DTU] Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What can I help with?"):
    # User message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Bot message
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response, _ = router.route_query(prompt)
                st.markdown(response)
            except Exception as e:
                response = f"❌ Error: {e}"
                st.error(response)
    st.session_state.messages.append({"role": "assistant", "content" : response})

