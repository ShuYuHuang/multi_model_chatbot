# For Streamlit
import streamlit as st
# from streamlit_chat import message


# For tackling uploaded files
# import tempfile
from glob import glob
import os
from os import path

# For index deletion
import requests

# For file retrieval
from client_es import ElasticsearchIndexer
from file_processor import FileProcessor
from params import (
    DEFAULT_LLM_ARGS,
    DEFAULT_TMP_DIR,
    ALL_FILE_TYPES,
    DOCUMENT_FILE_TYPES,
    SHEET_FILE_TYPES,
    DEFAULT_INDEX_NAME,
    DEFAULT_ES_URL
)

# For chatting integration
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


QUERY_INTEGRATION_TEMPLATE  = """
Request:

The questions are all related to either query results or history, 
please reply to the asked questions only, do not extend to other questions

If the question is related to querying results:
- Extract information from queried result and answer the question
Else if the question is related to history:
- Extract information from history and answer the question

Queried results form database:
```
{queried}
```

History:
```
{history}
```

You are asked:
{input}

Respond:
"""

# If the question is not related to either querying result or history:
# - Politely inform that the result is not related to either history or database
# """
query_integration_prompt = PromptTemplate(
        template=QUERY_INTEGRATION_TEMPLATE,
        input_variables=["history", "input", "queried"]
)

# Section for uploading files
def add_files_section():
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    with st.sidebar.expander("➕ &nbsp; Add Files", expanded=True):
        st.button("Clear Conversation & Data 🗑", on_click=delete_indices)
        uploaded_files = st.file_uploader(
            "Upload File",
            type=['csv', 'pdf', 'txt'],
            # type=ALL_FILE_TYPES,
            accept_multiple_files=True,
            key=st.session_state["file_uploader_key"])
        
        return uploaded_files

# Section for upload file to elastic search
def upload_file_to_esearch(uploaded_files, es_indexer):

    # Preventing uploading multiple times
    if uploaded_files is not []:
        for file in uploaded_files:
            # Parse file name
            fname = file.name.split('/')[-1]
            _, suffix = fname.split(".")
            local_file_name = path.join(DEFAULT_TMP_DIR, fname)
            
            if not( local_file_name in st.session_state['upload_list'] ):
                # write binary to local file
                with open(local_file_name, 'wb') as temp_file:
                    temp_file.write(file.getvalue())
                es_indexer.index_files(local_file_name)
                st.session_state['upload_list'].append(local_file_name)
        return local_file_name, suffix
    else:
        return None, None


# Section for upload file to elastic search
def display_user_input_form():
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Enter:", placeholder="Talk to the data 👉 (:", key='input')
            submit_button = st.form_submit_button(label='Send')
    
    return response_container, user_input, submit_button

# Get reply from bot
def handle_user_input(conversation_chain, user_input, indexer):
    if 'history' not in st.session_state:
            st.session_state['history'] = []
    queried_docs = indexer.search_files(user_input, k=3)

    response = conversation_chain.predict(
        input=user_input,
        history="\n".join(map(lambda x: f"user:{x[0]} | robot:{x[1]}",st.session_state['history'])),
        queried = "\n".join(map(lambda x: x.page_content, queried_docs))
    )

    # response = conversation_chain.run(
    #     input_documents=queried_docs,
    #     question=user_input
    # )

    st.session_state['history'].append((user_input, response))
    return response

# Final display on chat container
def display_messages():

    # Initial messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me about the uploaded files 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # Chat history
    with response_container:
        for i in range(len(st.session_state['generated'])):
            with st.chat_message(name="user",avatar="🧑‍💻"):
                st.markdown(st.session_state["past"][i])
            with st.chat_message(name="robot",avatar="🤖"):
                st.markdown(st.session_state["generated"][i])

def delete_indices():
    if ("es_url" in st.session_state) and ("index_name" in st.session_state):
        # Delete indices
        response = requests.delete(f'{st.session_state["es_url"]}/{st.session_state["index_name"]}')
        # Delete local files
        for f_name in st.session_state['upload_list']:
            os.remove(f_name)
        print("deletion status:",response.text)
    st.session_state["file_uploader_key"] += 1
    if 'history' in st.session_state:
        del st.session_state['generated'], st.session_state['past'], st.session_state["history"]
    st.rerun()

def delete_conversations():
    if 'history' in st.session_state:
        del st.session_state['generated'], st.session_state['past'], st.session_state["history"]
    st.rerun()


# @st.cache_data(max_entries=1)
# def init_indices():
#     if ("es_url" in st.session_state) and ("index_name" in st.session_state):
#         response = requests.delete(f'{st.session_state["es_url"]}/{st.session_state["index_name"]}')
#     print("deletion status:",response.text)
#     return None


if __name__ == '__main__':
    from langchain.chains import LLMChain
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chat_models import ChatOpenAI
    # from client_vicuna import VicunaLLM

    import dotenv
    # load system variables
    dotenv.load_dotenv()

    LLM_CLASS = ChatOpenAI
    # LLM_CLASS = VicunaLLM
    processor = FileProcessor(llm_class=LLM_CLASS, llm_args=DEFAULT_LLM_ARGS)

    # Create an Elasticsearch indexer.
    elasticsearch_indexer = ElasticsearchIndexer(
        llm=LLM_CLASS,
        embedding=HuggingFaceEmbeddings,
        index_name=DEFAULT_INDEX_NAME,
        es_url=DEFAULT_ES_URL
    )
    elasticsearch_indexer.add_processor(processor)

    # Create LLM
    llm = LLM_CLASS(**DEFAULT_LLM_ARGS)

    st.session_state["es_url"] = elasticsearch_indexer.es_url
    st.session_state["index_name"] = elasticsearch_indexer.index_name
    
    # Clear the indices at first time

    st.title("MMChat 📄🎥📢🦜🦙")
    # Upload file
    uploaded_files = add_files_section()

    if 'upload_list' not in st.session_state:
        st.session_state['upload_list'] = glob(path.join(DEFAULT_TMP_DIR, '*.*'))

    if len(uploaded_files)>0:
        
        # Upload button and mechanisms
        local_file_name, suffix = upload_file_to_esearch(uploaded_files, elasticsearch_indexer)

        # Create conversation chain
        # conversation_chain = load_qa_chain(
        #     llm=llm,
        #     verbose=True,
        # )

        conversation_chain = LLMChain(
            llm=llm,
            verbose=True,
            prompt=query_integration_prompt,
        )

        # Delete Button
        st.button("New Conversation 🆕", on_click=delete_conversations)

        response_container, user_input, submit_button = display_user_input_form()

        if submit_button and (user_input is not None):
            # Handle user input and generate a response
            response = handle_user_input(conversation_chain, user_input, elasticsearch_indexer)

            # Update session state
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(response)

        # Display messages
        display_messages()

    ######################################