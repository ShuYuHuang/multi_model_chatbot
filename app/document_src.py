import streamlit as st

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.embeddings.openai import OpenAIEmbeddings

from params import DOCUMENT_FILE_TYPES, SHEET_FILE_TYPES

import tempfile
import dotenv
dotenv.load_dotenv()

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0) # can be pre-loaded

@st.cache_resource()
def store_doc(uploaded_file, index_name="text-index"):
    suffix = uploaded_file.split(".")[-1]
    assert suffix in DOCUMENT_FILE_TYPES
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
        # Save a temporary file
        file_contents = uploaded_file.getvalue()
        
        tmp_file.write(file_contents)
        tmp_file_path = tmp_file.name

        loader = TextLoader(tmp_file_path)