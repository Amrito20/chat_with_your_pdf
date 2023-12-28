import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai 
from PyPDF2 import PdfReader

#configuring the Palm api for the LLM 
api_key = st.secrets["Google_API_Key"]  
genai.configure(api_key=api_key)
llm = GooglePalm(google_api_key= api_key, temperature= 0.1)


#get the texts of the pdf
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

#creating chunks of the texts
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

#creating embeddings and vectorstore
def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

#Retrieve the documet for conversation
def get_conversational_chain(vector_store):
    llm = GooglePalm(google_api_key= api_key, temperature= 0.1)
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               retriever=vector_store.as_retriever(), 
                                                               memory=memory)
    return conversation_chain

#Response generating on the UI with chat history
def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i%2 == 0:
            st.write("**YOU:**  \n", message.content)
        else:
            st.write("***<<<BOT>>>*** \n\n", message.content)
            st.write("---------------------------")


def main():
    st.set_page_config(page_title="caht with your document...",
                       page_icon=":books",
                       menu_items=None,
                       initial_sidebar_state="expanded")
    
    # Load the CSS using st.markdown
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # setting the UI:
    st.header("Chat with your PDF:books:")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None

    #User question set    
    if user_question:
        user_input(user_question)
    
    #siderbar to upload files
    with st.sidebar:
        st.subheader("Upload your files and click ''***Process***''")
        pdf_docs = st.file_uploader("", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                #get the text from the pdf:
                raw_text = get_pdf_text(pdf_docs)

                #creating chunks of the documents:
                text_chunks = get_text_chunks(raw_text)

                #creating Embeddings and putting in vertorestore
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Done")


if __name__ == "__main__":
    main()
