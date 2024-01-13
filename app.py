import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers, HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


DB_FAISS_PATH = 'vectorstore/db_faiss'

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    #llama-2-7b-chat.ggmlv3.q8_0.bin
    llm = HuggingFaceHub(
        repo_id = "google/flan-t5-xxl",
        model_kwargs={
        "max_length": 512,
        "temperature": 0.5
        }
    )
    return llm
def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text

def get_text_chunks(raw_text):
  text_splitter = CharacterTextSplitter(
    separator="\n", 
    chunk_size=10000,
    chunk_overlap=1000,
    length_function=len
    )
  chunks = text_splitter.split_text(raw_text)
  return chunks

def get_gemini_model():
  return genai.GenerativeModel("gemini-pro")
def get_gemini_response(question):
  model=get_gemini_model()
  response = model.generate_content(question)
  return response.text
  
  

def get_vector_store(text_chunks):
  #embeddings = OpenAIEmbeddings()
  #'dmis-lab/biobert-base-cased-v1.1-mnli'
  #'model_name='hkunlp/instructor-xl'
  #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
  #embeddings = genai.embed_content(content=text_chunks, model='models/embedding-001',task_type="retrieval_document")
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  #print(embeddings)
  vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
  vectorstore.save_local(DB_FAISS_PATH)
  return vectorstore

def get_conversation_chain():
  #llm = get_gemini_model() #load_llm()
  prompt_template = """
  Answer the question as detailed as possible from provided context, make sure to provide all details, if the answer is not available in context, just say "answer is not available in the context", don't provide the wrong answer.
  Context:\n {context}?\n
  Question:\n {question}\n
  
  Answer:
  """
  model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
  prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])
  coversation_chain = load_qa_chain(model, chain_type="stuff",prompt=prompt)
  
  #memory = ConversationBufferMemory(memory_key='chat_history', return_message=True, output_key='answer')
  #coversation_chain = ConversationalRetrievalChain.from_llm(
  #  llm = llm, 
  #  retriever = vectorstore.as_retriever(),
  #  memory = memory,
  #  return_source_documents=True, 
  #  get_chat_history=lambda h : h
  #)
  return coversation_chain

def handle_userinput(user_question):
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  new_db = FAISS.load_local(DB_FAISS_PATH, embeddings)
  docs = new_db.similarity_search(user_question)
  chain = get_conversation_chain()
  
  response = chain(
    {"input_documents":docs, "question": user_question}
    , return_only_outputs=True
  ) 
  #response = st.session_state.conversation({'question':user_question})
  #st.write(response)
  #st.session_state.chat_history = response['chat_history']
  #strng = st.session_state.chat_history
  #strng = strng.replace("AI:",",AI:").replace("Human:",",Human:").split(",")[1:]
  st.write(user_template.replace("{{MSG}}","Human: "+user_question), unsafe_allow_html=True)
  st.write(bot_template.replace("{{MSG}}","AI: "+response['output_text']), unsafe_allow_html=True)
  #for i, message in enumerate(strng):
  #  if i % 2 == 0:
      #st.write(message)
  #    st.write(user_template.replace("{{MSG}}",str(message)), unsafe_allow_html=True)
  #  else:
      #st.write(message)
  #    st.write(bot_template.replace("{{MSG}}",str(message)), unsafe_allow_html=True)
  #st.write(bot_template.replace("{{MSG}}",response['answer']), unsafe_allow_html=True)
def main():
  load_dotenv()
  genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
  st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
  
  st.write(css, unsafe_allow_html=True)
  
  #st.write(st.session_state)
  
  if "conversation" not in st.session_state:
    st.session_state.conversation = None
  
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
  
  st.header("Chat with Multiple PDFs :books:")
  user_question = st.text_input("Ask a question about your documents:")
  if user_question:
    handle_userinput(user_question)
  
  #st.write(user_template.replace("{{MSG}}","Hello robot"), unsafe_allow_html=True)
  #st.write(bot_template.replace("{{MSG}}","Hello human"), unsafe_allow_html=True)
  
  with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
      with st.spinner("Processing"):                
        # get pdf text from
        raw_text = get_pdf_text(pdf_docs)
        # get the text chunks
        text_chunks = get_text_chunks(raw_text)
        st.write(text_chunks)
        # create vector store
        get_vector_store(text_chunks)
        st.success("Done")
        
        # create conversation chain
        #st.session_state.conversation = get_conversation_chain(vectorstore)
        
  #st.session_state.conversation  
if __name__=='__main__':
  main()