# -Import
## For Streamlit
import streamlit as st
# from streamlit_chat import message


## For tackling uploaded files
# import tempfile
from glob import glob
import os
from os import path

## For index deletion
import requests

## For file retrieval
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

## For chatting integration

from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent

## For plotting and plot summary
import re
import base64
import json
from client_caption import LlavaCaptioner, LLAVA_URL1
import pandas as pd
import plotly.io as pio
import plotly.express as px





# from prompt_template import (
from prompt_template_ar import (
    QUERY_INTEGRATION_TEMPLATE,
    DATA_SHORT_DESCRIPTION,
    PRMPTED_CSV_PLOT,
    DISCRIBE_PLOT,
    INITIAL_PLOT_INSTRUCTION

)


# -Prompt templates


# prmpted_csv_plot_prompt = PromptTemplate(
#         template=PRMPTED_CSV_PLOT,
#         input_variables=["filename", "head3lines", "instructions"]
# )

# describe_plot_prompt = PromptTemplate(
#         template=PRMPTED_CSV_PLOT,
#         input_variables=[""]
# )

# -Tools
## For code generation
def get_image_code(response_text):
    python_pattern = r'```python\s(.*?)```'
    csv_pattern = r"\ncsv_df\s*=\s*.+\n"

    # Use re.match to replace the matched pattern with an empty string
    matches = re.findall(python_pattern, response_text, re.DOTALL)
    if not matches:
        return ""
    code =  matches[0]
    code = re.sub(csv_pattern, "", code)
    code = code.replace("fig.show()", "")
    return code

# -Section for uploading files
## add files to system
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

# -Retrieval and reply
## QA prompt with history and query
query_integration_prompt = PromptTemplate(
    template=QUERY_INTEGRATION_TEMPLATE,
    input_variables=["history", "input", "queried"]
)

def handle_user_input(chain, indexer, user_input):
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    queried_docs = indexer.search_files(user_input, k=3)

    response = chain.predict(
        input=user_input,
        history="\n".join(map(lambda x: f"user:{x[0]} | robot:{x[1]}",st.session_state['history'])),
        queried = "\n".join(map(lambda x: x.page_content, queried_docs))
    )
    # response = chain.run(
    #     input_documents=queried_docs,
    #     question=user_input
    # )
    st.session_state['history'].append((user_input, response))
    return response

# -Plot functions
## decode the responded json format
def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)
    

def plot_with_analysis(
        llm,
        captioner,
        local_file,
        user_input):
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    # queried_docs = indexer.search_files(user_input, k=1)

    csv_df = pd.read_csv(local_file)

    agent = create_pandas_dataframe_agent(
        llm,
        csv_df,
        verbose=True)

    response = agent.run(
        PRMPTED_CSV_PLOT.format(
            file_name=local_file,
            instructions=user_input
        )
    )
    
    response_dict = decode_response(str(response))

    ## Write a response from an agent to a Streamlit app.
    fig = None
    if "answer" in response_dict:
        final_response = response_dict["answer"]
        return final_response, fig

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        final_response = df.to_markdown()
        return final_response, fig
    
    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            df_data = {
                    col: [x[i] if isinstance(x, list) else x for x in data['data']]
                    for i, col in enumerate(data['columns'])
                }       
            df = pd.DataFrame(df_data)
            fig = px.bar(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df_data = {col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])}
            df = pd.DataFrame(df_data)
            # Create a figure and axis using pyplot
            fig = px.line(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")
    
    image_bytes = pio.to_image(fig, format="png")   
    base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    import pickle
    pickle.dump(base64_encoded_image, open('tmp.pkl', 'wb'))


    final_response = captioner.send_frame(
        base64_encoded_image,
        DISCRIBE_PLOT.format(user_input=user_input)
    )

    st.session_state['history'].append((user_input, final_response))
    return final_response, fig

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
    captioner = LlavaCaptioner(LLAVA_URL1)


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

    conversation_chain = LLMChain(
        llm=llm,
        verbose=True,
        prompt=query_integration_prompt,
    )

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



        # Delete Button
        st.button("New Conversation 🆕", on_click=delete_conversations)

        response_container, user_input, submit_button = display_user_input_form()

        if ('history' not in st.session_state) and (suffix == 'csv'):
            # Display images if there is images
            with st.container() as analytic_sum_page:
                response, myfig = plot_with_analysis(
                    llm,
                    captioner,
                    local_file_name,
                    INITIAL_PLOT_INSTRUCTION
                )
                if myfig:
                    st.session_state['fig'] = myfig

            st.session_state['past'] = ['Summary:']
            st.session_state['generated'] = [response]


        if submit_button and (user_input is not None):
            # Handle user input and generate a response
            if suffix == 'csv':
                with st.container() as analytic_sum_page:
                    response, myfig = plot_with_analysis(
                        llm,
                        captioner,
                        local_file_name,
                        user_input)
                    if myfig:
                        st.session_state['fig'] = myfig
                # Update session state
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(response)
            else:
                response = handle_user_input(
                    conversation_chain,
                    elasticsearch_indexer,
                    user_input)
            
                # Update session state
                st.session_state['past'].append(user_input)
            st.session_state['generated'].append(response)
        # Display images if there is images
        with st.container() as visul_display:
            with st.expander("Visulaization", expanded=True):
                if 'fig' in st.session_state:
                    st.plotly_chart(st.session_state['fig'], theme='streamlit', use_container_width=True)
        # Display messages
        display_messages()

        

    ######################################
