# Supported File Types
DOCUMENT_FILE_TYPES = ["txt", "pdf", "docx"]
SHEET_FILE_TYPES = ["csv", "xls", "xlsx"]
MEDIA_FILE_TYPES = ["mp4", "avi", "mov", "mkv", "mp3", "wav","m4a"]
IMAGE_FILE_TYPES = ["jpg", "jpeg", "png", "bmp", "tiff"]
ALL_FILE_TYPES = DOCUMENT_FILE_TYPES + SHEET_FILE_TYPES + MEDIA_FILE_TYPES + IMAGE_FILE_TYPES

# DEFAULT_LLM_ARGS = dict(
#     model_name='gpt-3.5-turbo',
#     temperature=0,
#     max_tokens=1000
# )

DEFAULT_LLM_ARGS = dict(
    model_name='vicuna-13b-v1.5-16k',
    temperature=0.0,
    max_tokens=512
)

DEFAULT_INDEX_NAME = "myindex"
DEFAULT_ES_URL = "http://localhost:9200"
DEFAULT_TMP_DIR = "tmp"

DEFAULT_MODEL_DIR = "../models"