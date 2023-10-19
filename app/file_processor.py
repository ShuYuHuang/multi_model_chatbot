import tempfile
import requests

# from PIL import Image
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
# from model import WhisperModel, BLIPModel

# for amount control
import streamlit as st

# get parameters
from params import (
    DEFAULT_LLM_ARGS,
    DEFAULT_TMP_DIR,
    MEDIA_FILE_TYPES,
    IMAGE_FILE_TYPES,
)
# load system variables
import dotenv
dotenv.load_dotenv()


def extract_from_transcript(segments):

    doc = "\n".join([f"{seg['text']} (00:{seg['start']:.2f})" for seg in  segments])

    return doc

DIARIZER_TEMPLATE = """
Determine the characters for the transcript in SRT format, and assign the character to the beginning of transcript, e.g. 
99
00:32:01.000 --> 00 00:32:02.000
Speaker C: The suspect didn't show up last night

Here is the input:
{transcript}

Return:
"""
diarizer_prompt = PromptTemplate(
    input_variables=["transcript"],
    template=DIARIZER_TEMPLATE
)


WHISPER_URL = 'http://10.100.100.106:8001/v1/audio/transcriptions'
WHISPER_HEADERR = {
    'accept': 'application/json',
    'Content-Type': 'application/x-www-form-urlencoded'
}

def transcribe(uploaded_file, whisper_model):
    data = {
        'existed_file': uploaded_file,
        'model': whisper_model,
        'response_format': 'srt',
        'temperature': '0',
    }
    return requests.post(WHISPER_URL, headers=WHISPER_HEADERR, data=data).json()

def split_list(original_list, length):
    sublists = [original_list[i:i + length] for i in range(0, len(original_list), length)]
    return sublists

@st.cache_resource(max_entries=1)
class FileProcessor:
    def __init__(self,
                 llm_class=ChatOpenAI,
                 llm_args=DEFAULT_LLM_ARGS,
                 model_root='models',
                 whisper_model="large-v2",
                 initialize_prompt=None,
                 diarizer_prompt=diarizer_prompt,
                 chunk_size=1000,
                 chunk_overlap=0):
        
        # Whisper parameters
        self.whisper_model = whisper_model
        self.model_root = model_root
        self.initialize_prompt = initialize_prompt
        self.diarizer_prompt = diarizer_prompt
        self.llm_args = llm_args
        
        self.llm = llm_class(**self.llm_args)

        self.transcript_diarizer = LLMChain(
            llm=self.llm,
            prompt=diarizer_prompt
        )

        # Splitters
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)
    
    def process_file(self, uploaded_file):
        # Get the file type.
        file_type = uploaded_file.split(".")[-1]
        # Process the file based on the file type.
        if file_type == 'pdf':
            return self.extract_text_from_pdf(uploaded_file)
        elif file_type in ['txt', 'docx']:
            return self.extract_text_from_text_file(uploaded_file)
        elif file_type in MEDIA_FILE_TYPES:
            return self.extract_text_from_speech_audio(uploaded_file)
        # elif file_type in ['jpg', 'png']:
        #     return self.extract_features_from_image(uploaded_file)
        else:
            raise ValueError('Unsupported file type: {}'.format(file_type))

    def extract_text_from_pdf(self, uploaded_file):
        # Load the PDF file.
        loader = PyPDFLoader(uploaded_file)
        docs = loader.load_and_split(self.splitter)
        return docs

    def extract_text_from_text_file(self, uploaded_file):
        # Load the text file.
        loader = TextLoader(uploaded_file)
        docs = loader.load_and_split(self.splitter)
        return docs

    
    def extract_text_from_speech_audio(self, uploaded_file):
        
        conversations = transcribe(uploaded_file, whisper_model=self.whisper_model)
        
        doc= []
        data = self.transcript_diarizer.run(transcript=conversations)
        doc.append(Document(page_content=data, metadata={"source": uploaded_file}))

        docs = self.splitter.split_documents(doc)
        # Determine the speaker

        return docs

    # def extract_features_from_image(self, uploaded_file):
    #     # Load the image file.
    #     with io.BytesIO(uploaded_file.read()) as image_file:
    #         image = Image.open(image_file)

    #     # Extract the features from the image using BLIP.
    #     features = self.blip_model.extract_features(image)

    #     return features



if __name__ == '__main__':
    # Create a file processor.
    import streamlit as st
    import tempfile
    import json
    file_processor = FileProcessor(
        whisper_model='large-v2'
    )

    # Process a sample file.
    uploaded_files = st.file_uploader(
        "Choose a file",
        accept_multiple_files=True,
        key="mykey")
    if uploaded_files is not []:

        for files in uploaded_files:
            suffix = "."+files.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=DEFAULT_TMP_DIR) as tmp_file:
                # Save a temporary file
                tmp_file.write(files.getvalue())
                # Extract contents from the file
            text = file_processor.process_file(tmp_file.name)
            for t in text:
                st.markdown(t.page_content)