# General settings
PROJECT_NAME = "X_LeBench_dataset_construction"
VERSION = "1.0.0"
CHUNK_NUM = 16   # We generate 16 chunks, and set different selecting number (4,9,15) to get lifelogs of diverse lenth.
IOU_THRESHOLD = 0.33  # IoU threshold of scenarios for retrieval
LOG_LENTHS = [4, 9, 15]  # Must be list[int] and no more than CHUNK_NUM

# Paths
EGO4DINFO_PATH = "generation/ego4d_info"
MEMORY_PATH = "path/to/save/memory"
PROMPT_PATH = "generation/prompt_template"
PERSONA_PATH = "path/to/save/persona"
SAVE_PERSONA_LIST = "persona_ids.json"
SAVE_MEMORY_LIST = "memory_list.json"

# Generate tools, the str should be in ["openai","gemini","minicpm", "qwen"]
GENERATE_WAY = "openai"
GEN_MODEL = "gpt-4o"   #"gpt-4o" "gemini-1.5-pro" "qwen-max-0919"
MODEL_TEMPERATURE = 0.7

# API keys 
GEMINI_API_KEY = ""   # Replace it with your API key
OPENAI_API_KEY = ""
QWEN_API_KEY =  ""