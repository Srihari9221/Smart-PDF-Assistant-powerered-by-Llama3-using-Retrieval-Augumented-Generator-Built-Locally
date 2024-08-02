import os
import pickle
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

# Load environment variables
load_dotenv()

# Set model and initialize Ollama
MODEL = "llama3"
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# Initialize components
parser = StrOutputParser()

# Define a prompt template with a specific prompt message
template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know".

Please enter your question:
Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# File path to store serialized pages
pages_file = "pages.pkl"

# Function to load or vectorize pages
def load_or_vectorize_pages():
    if os.path.exists(pages_file):
        with open(pages_file, 'rb') as f:
            pages = pickle.load(f)
        print("Loaded pages from file.")
    else:
        # Load PDF data if not already loaded
        pdf_file = "OSS_Stack_2022_v8.8_2022-12-01.pdf"
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"PDF file '{pdf_file}' not found.")

        loader = PyPDFLoader(pdf_file)
        pages = loader.load_and_split()

        # Vectorize and save pages to file
        vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
        with open(pages_file, 'wb') as f:
            pickle.dump(pages, f)
        print(f"Saved vectorized pages to {pages_file}.")

    return pages

# Load or vectorize pages
pages = load_or_vectorize_pages()

# Create vector store from pre-loaded vectorized documents
vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Processing loop to handle user questions
while True:
    # Prompt user for question input
    user_question = input("Enter your question (or 'exit' to quit): ").strip()
    if user_question.lower() == "exit":
        break

    # Set up the processing chain for question answering
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | parser
    )

    # Invoke the chain to get the answer
    answer = chain.invoke({"context": "Here is some context", "question": user_question})
    print(f"Answer: {answer}")
