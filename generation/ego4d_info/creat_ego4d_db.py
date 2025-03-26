import json
from sentence_transformers import SentenceTransformer
import chromadb
import sys
import os
import generation.config as config


# Initialize embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load Ego4d video info data
sum_path = os.path.join(config.EGO4DINFO_PATH, "video_info.json")
with open(sum_path, 'r') as f:
    data = json.load(f)

# Embedding the summaries
for video in data:
    summary = video['consolidated_summary']
    embedding = model.encode(summary).tolist()
    video["sum_embedding"] = embedding


# Create chroma database
client = chromadb.PersistentClient(os.path.join(config.EGO4DINFO_PATH, "chromaDB"))
collection = client.create_collection(name='video_info_all')

for item in data:
    meta = {key: value for key, value in item.items() if key not in ["sum_embedding", "video_scenarios"]}
    collection.add(ids=item['video_uid'], embeddings=item['sum_embedding'], metadatas = meta, documents=item["consolidated_summary"])
