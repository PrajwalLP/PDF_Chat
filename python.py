import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmltemplates import css ,user_template, bot_template

def get_pdf_text(pdf_doc):
    text=""
    pdf_reader=PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

def get_vectorstore(chunk):
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_texts(texts=chunk , embedding = embeddings)
    return vector

def get_conversation(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history" , return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory= memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.converse({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate (st.session_state.chat_history):
        if i %2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else :
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main ():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF" , page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "converse" not in st.session_state:
        st.session_state.convere = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        

    st.header("Chat with PDF:books:")
    user_question  = st.text_input("Ask a question :")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello Bot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your PDF")
        pdf_doc = st.file_uploader("Upload your PDF here and click on process")
        if st.button("process"):
            with st.spinner("Processing"):
                #Getting text from PDF
                text_raw = get_pdf_text(pdf_doc)

                #Making small Chunks from the PDF
                chunks = get_chunks(text_raw)

                #Feed the Chunks inot the Backend 
                vectorstore = get_vectorstore(chunks)

                #retain the History and Let the Bot converse throuh it 
                st.session_state.converse = get_conversation(vectorstore)

if __name__=='__main__':
    main()
