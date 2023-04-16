import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PIL import Image

image = Image.open('synacal_logo.png')

# def set_key(key):
#     if 'key' not in st.session_state:
#         st.session_state['key'] = key
#         st.write(st.session_state['key'])
#         os.environ["OPENAI_API_KEY"] = st.session_state['key']

# def bot(path, query):
#     set_key("YOUR_OPENAI_API_KEY")
#     embeddings = OpenAIEmbeddings()
#     reader = PdfReader(path)
#     raw_text = ''
#     for i, page in enumerate(reader.pages):
#         text = page.extract_text()
#         if text:
#             raw_text += text
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#     )
#     texts = text_splitter.split_text(raw_text)
#     docsearch = FAISS.from_texts(texts, embeddings)
#     chain = load_qa_chain(OpenAI(), chain_type="stuff")
#     docs = docsearch.similarity_search(query)
#     answer = chain.run(input_documents=docs, question=query)
#     return answer

# def main():
#     st.title("Doxplore")
#     with st.sidebar:
#         st.image(image)
#         st.write('')
#         api_key = st.text_input("Enter your OpenAI API key:")
#         set_key_button = st.button("Set key", on_click=set_key(api_key))
#         uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
#     if api_key and uploaded_file:
#         query = st.text_input("Enter your question:")
#         get_answer_button = st.button("Get answer", on_click=bot(uploaded_file, query))
#         delete_key_button = st.button("Delete key", on_click=os.environ.pop("OPENAI_API_KEY"))
#         delete_pdf_button = st.button("Delete PDF", on_click=uploaded_file.close())
#     else:
#         st.write("Chat with your document!")
    

# if __name__ == "__main__":
#     main()

# import os
# import streamlit as st
# from py_pdf_parser import PdfReader
# from faiss import FAISS
# from openai.encoder import OpenAIEmbeddings
# from openai.completion import Completion

# Set the OpenAI API key
def set_key(key):
    os.environ["OPENAI_API_KEY"] = key

# Load the QA chain
# def load_qa_chain(api_client, chain_type):
#     return Completion.create(
#         engine="davinci" if chain_type == "stuff" else "curie",
#         prompt=api_client.api_key,
#         temperature=0.5,
#         max_tokens=1024,
#         top_p=1.0,
#         frequency_penalty=0,
#         presence_penalty=0,
#     )

# Delete the API key
def delete_api_key():
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

# Delete the uploaded PDF file
def delete_pdf_files():
    if "pdf_path" in st.session_state:
        os.remove(st.session_state.pdf_path)

# Initialize the state variables
def init_state():
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None

# Define the bot function
def bot(query):
    embeddings = OpenAIEmbeddings()
    reader = PdfReader(st.session_state.pdf_path)
    raw_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    docsearch = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)
    return answer

# Define the app
def app():
    init_state()

    st.title("DOXPLORE")
    st.write('Introducing Synacal\'s latest SaaS product, an AI-powered PDF question answering system! With our cutting-edge technology,you can now easily extract information from any PDF document by simply asking a question. Whether you need to quickly find a specific piece of information or want to better understand the contents of a document, our system is here to help. Try it out today and experience the power of AI!')
    st.sidebar.image(image)
    # Get the API key
    key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    set_key(key)

    # Upload the PDF
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        # Save the PDF to a temporary file
        st.session_state.pdf_path = f"uploaded_pdf.pdf"
        with open(st.session_state.pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    if not st.session_state.pdf_path:
        st.warning("Please upload a PDF to continue.")
        return
    st.sidebar.write(f"Current PDF: {os.path.basename(st.session_state.pdf_path)}")

    # Ask the question
    query = st.text_input("Ask a question")
    if not query:
        return

    # Get the answer
    answer = bot(query)
    st.write("Answer:", answer)

    # Delete the API key and PDF files
    if st.button("Delete API key and PDF"):
        delete_api_key()
        delete_pdf_files()

if __name__ == "__main__":
    app()

