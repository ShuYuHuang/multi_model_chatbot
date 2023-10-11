# For Streamlit
import streamlit as st
from streamlit_chat import message
from params import (
    DOCUMENT_FILE_TYPES,
    SHEET_FILE_TYPES,
    MEDIA_FILE_TYPES,
    IMAGE_FILE_TYPES,
    ALL_FILE_TYPES
)

# For tackling uploaded files
import tempfile


# Sections to use

def add_files_section():
    with st.sidebar.expander("➕ &nbsp; Add Files", expanded=True):
        uploaded_files = st.file_uploader("Upload File", type=ALL_FILE_TYPES, accept_multiple_files=True)
        return uploaded_files

def upload_file_to_esearch(uploaded_files):
    if len(uploaded_files)==1:
        if uploaded_files.split(".")[-1] in DOCUMENT_FILE_TYPES:
            
            db = store_docs(uploaded_files, index_name="paragraph-index")
    else:
        for uploaded_file in uploaded_files:
            
    return db

def display_user_input_form():
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Enter:", placeholder="Talk to the data 👉 (:", key='input')
            submit_button = st.form_submit_button(label='Send')
    
    return response_container, user_input, submit_button

def initialize_messages():
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me about the uploaded files 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]


def handle_user_input(user_input):
    # Logic to generate a response based on user input
    response = "Response to: " + user_input
    return response

def display_messages():
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + '_user',
                    avatar_style="big-smile")
                message(
                    st.session_state["generated"][i],
                    key=str(i),
                    avatar_style="thumbs")

def store_to_elasticsearch

if __name__ == '__main__':
    #### Execution flow of the code #####
    uploaded_files = add_files_section()
    print(uploaded_files)
    if uploaded_files:
        

        


        response_container, user_input, submit_button = display_user_input_form()

        initialize_messages()

        if submit_button and user_input:
            # Handle user input and generate a response
            response = handle_user_input(user_input)

            # Update session state
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(response)

        # Display messages
        display_messages()

    ######################################