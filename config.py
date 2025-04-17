# config.py
import os

def get_openai_api_key():
    # Priority: Environment var > Streamlit secrets > Fallback
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")

    try:
        import streamlit as st
        return st.secrets["OPENAI_API_KEY"]
    except (ImportError, KeyError):
        pass  # Streamlit not available or secret missing

    raise RuntimeError("OPENAI_API_KEY not found in env or secrets.")
