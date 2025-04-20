import argparse
import weaviate
import os
from openai import OpenAI
from weaviate.classes.init import Auth

# ---------------------------
# Vector DB configurations
# ---------------------------
vector_db_configs = {
    "programme_db": {
        "url": "g9h6eircsbe9d9doiv1w.c0.europe-west3.gcp.weaviate.cloud",
        "api_key": "A6IFGm879tMitR94FWoffvNsBDeMTu8eZv8n",
        "collection_name": "Chunk"
    },
    "course_db": {
        "url": "https://yz34awbrqlko1tvblm77g.c0.europe-west3.gcp.weaviate.cloud",
        "api_key": "YVldI0WBz6MUZVoPZA5wp5t7zalMI12jdkfm",
        "collection_name": "Course"
    }
}

# ---------------------------
# Command-line argument setup
# ---------------------------
parser = argparse.ArgumentParser(description="Delete specific Weaviate collections.")
parser.add_argument("--delete-course", action="store_true", help="Delete the 'Course' collection from course_db")
parser.add_argument("--delete-chunk", action="store_true", help="Delete the 'Chunk' collection from programme_db")
args = parser.parse_args()

# ---------------------------
# Set OpenAI key
# ---------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# ---------------------------
# Function to connect and delete a collection
# ---------------------------
def delete_collection(db_config):
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=db_config["url"],
            auth_credentials=Auth.api_key(db_config["api_key"]),
            headers={"X-OpenAI-Api-key": openai_key}
        )
        client.collections.delete(db_config["collection_name"])
        print(f"Collection '{db_config['collection_name']}' deleted from {db_config['url']}")
        client.close()
    except Exception as e:
        print(f"Error deleting '{db_config['collection_name']}' from {db_config['url']}: {e}")

# ---------------------------
# Deletion logic based on user input
# ---------------------------
if args.delete_course:
    delete_collection(vector_db_configs["course_db"])

if args.delete_chunk:
    delete_collection(vector_db_configs["programme_db"])
