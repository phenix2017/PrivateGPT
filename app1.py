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

def get_text_from_pdf(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
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

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def setup_chain(knowledge_base, user_question):
    docs = knowledge_base.similarity_search(user_question)
    llm = OpenAI()
    from langchain.chains.question_answering import load_qa_chain
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain, docs

def handle_userinput(user_question):
    response = streamlit.session_state.conversation({'question': user_question})
    streamlit.session_state.chat_history = response['chat_history']

    # for i, message in enumerate(streamlit.session_state.chat_history):
    #     if i % 2 == 0:
    #         streamlit.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         streamlit.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    streamlit.set_page_config(page_title="Ask your PDF")
    streamlit.header("Ask your PDF")
    user_question = streamlit.text_input("Ask a question about your PDF")
    # if user_question:
    #     handle_userinput(user_question)
    with streamlit.sidebar:
        pdfs = streamlit.file_uploader("Upload your PDF", accept_multiple_files=True)
        if streamlit.button("Process"):
            with streamlit.spinner("Processing"):
                text = get_text_from_pdf(pdfs)
                chunks = get_text_chunk(text=text)
                knowledge_base = get_vectorstore(text_chunks=chunks)
                if user_question:
                    chain, docs = setup_chain(knowledge_base=knowledge_base, user_question=user_question)
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=user_question)
                        print(cb)
                        streamlit.write(response)

if __name__ == '__main__':
    main()

    print("hello world")