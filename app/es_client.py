# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI

# Embedding and Language models
from langchain.embeddings import HuggingFaceEmbeddings
from vicuna_client import VicunaLLM

# Vector database stuffs
from langchain.vectorstores import ElasticsearchStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

import streamlit as st

from file_processor import FileProcessor
from params import (
    DEFAULT_ES_URL,
    DEFAULT_TMP_DIR,
    DEFAULT_INDEX_NAME,
    DEFAULT_MODEL_DIR
)

QUERY_TEMPLATE="""You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.
Original question: {question}"""
example_query_prompt = PromptTemplate(
    input_variables=["question"],
    template=QUERY_TEMPLATE,
)

class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

@st.cache_resource(max_entries=10)
class ElasticsearchIndexer:
    def __init__(self,
                #  embedding=OpenAIEmbeddings,
                #  llm=ChatOpenAI,
                 embedding=HuggingFaceEmbeddings,
                 llm=VicunaLLM,
                 query_prompt=example_query_prompt,
                 output_parser = LineListOutputParser(),
                 es_url=DEFAULT_ES_URL,
                 index_name=None,
                 distance_strategy="COSINE",
                 ):
        self.es_url = es_url
        self.index_name = index_name
        self.vdb = ElasticsearchStore(
            es_url = self.es_url,
            index_name=self.index_name,
            embedding=embedding(cache_folder=DEFAULT_MODEL_DIR),
            distance_strategy=distance_strategy
        )
        
        self.llm = llm(temperature=0)
        self.llm_chain = LLMChain(llm=self.llm, prompt=query_prompt, output_parser=output_parser)
        self.retriever = MultiQueryRetriever(
            retriever=self.vdb.as_retriever(),
            llm_chain=self.llm_chain,
            parser_key="lines"
        ) # "lines" is the key (attribute name) of the parsed output


    def index_files(self, uploaded_file, processor):
        # Extract the relevant information from the uploaded file.
        extracted_docs = processor.process_file(uploaded_file)

        # Index the extracted information in Elasticsearch.
        self.vdb.add_documents(extracted_docs)

    def search_files(self, query, k=3):
        self.vdb.client.indices.refresh(index=self.index_name)

        # Search for files in Elasticsearch based on the query.
        results = self.vdb.similarity_search(query=query, k=k)

        # Return the results.
        return results

if __name__ == '__main__':
    import streamlit as st
    import tempfile

    # Create Processor
    processor = FileProcessor()

    # Create an Elasticsearch indexer.
    elasticsearch_indexer = ElasticsearchIndexer(
        index_name=DEFAULT_INDEX_NAME,
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
            elasticsearch_indexer.index_files(tmp_file.name, processor)

            query = 'what is the document said'
            results = elasticsearch_indexer.search_files(query)

            st.write(results)