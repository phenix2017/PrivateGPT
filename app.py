from dotenv import load_dotenv
# import os 
import streamlit 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
DEBUG = False

def get_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunk(text=""):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
    )
    text_chunk = text_splitter.split_text(text)
    return text_chunk 

def get_embeddings_knowledage_base(text_chunk):
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunk, embeddings)
    return knowledge_base

def setup_chain(knowledge_base, user_question):
    docs = knowledge_base.similarity_search(user_question)
    llm = OpenAI()
    from langchain.chains.question_answering import load_qa_chain
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain, docs

def main():
    load_dotenv()
    streamlit.set_page_config(page_title="Ask your PDF")
    streamlit.header("Ask your PDF")
    pdf = streamlit.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        # pdf_reader = PdfReader(pdf)
        # text = ""
        # for page in pdf_reader.pages:
        #     text += page.extract_text()
        text = get_text_from_pdf(pdf)
        if DEBUG:
            streamlit.write(text)
        
        # text_splitter = CharacterTextSplitter(
        #     separator="\n",
        #     chunk_size=1000, 
        #     chunk_overlap=200,
        #     length_function=len,
        # )
        # chunks = text_splitter.split_text(text)
        chunks = get_text_chunk(text=text)
        if DEBUG:
            streamlit.write(chunks)
        # create embeddings 
        # knowledge_base = FAISS.from_texts(chunks, embeddings)
        knowledge_base = get_embeddings_knowledage_base(text_chunk=chunks)
        if DEBUG:
            streamlit.write(knowledge_base)
        # show user input
        user_question = streamlit.text_input("Ask a question about your PDF")
        if user_question:
            chain, docs = setup_chain(knowledge_base=knowledge_base, user_question=user_question)
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            streamlit.write(response)

if __name__ == '__main__':
    main()

    print("hello world")