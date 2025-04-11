import weaviate
import os
import json
import re
import pickle
import numpy as np
from openai import OpenAI
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


weaviate_client.collections.delete("Course")
print("Collection 'Course' deleted.")
weaviate_client.close()