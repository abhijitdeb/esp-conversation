from flask import Flask, request, jsonify, render_template
import logging
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from transformers import pipeline

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# Load the local NLP model (BART for summarization) using Hugging Face
try:
    nlp_model = pipeline("summarization", model="facebook/bart-large-cnn")
    logging.info("NLP model loaded successfully")
except Exception as e:
    logging.error(f"Error loading NLP model: {str(e)}")
    nlp_model = None

# Load and prepare the vector database
def load_vector_db():
    try:
        # Load your data (replace with your actual data loading logic)
        loader = TextLoader("path_to_your_data.txt")
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings()

        # Create vector store
        vector_db = FAISS.from_documents(texts, embeddings)

        logging.info("Vector database loaded successfully")
        return vector_db
    except Exception as e:
        logging.error(f"Error loading vector database: {str(e)}")
        return None

vector_db = load_vector_db()

# RAG-based chatbot function using the vector database
def rag_chatbot(user_query):
    logging.info(f"Processing user query: {user_query}")

    if vector_db is None:
        logging.error("Vector database not initialized")
        return "Sorry, there's an issue with the vector database. Please try again later."

    # Retrieve relevant documents
    docs = vector_db.similarity_search(user_query, k=3)

    # Create a prompt from the retrieved documents
    prompt = f"Based on the following information:\n\n"
    for doc in docs:
        prompt += f"{doc.page_content}\n\n"
    prompt += f"Answer the following question: {user_query}"

    logging.debug(f"Generated prompt: {prompt}")

    if nlp_model is None:
        logging.error("NLP model not initialized")
        return "Sorry, there's an issue with the NLP model. Please try again later."

    # Generate response
    generated_response = generate_response(prompt)
    logging.info(f"Generated response: {generated_response}")

    return generated_response

def generate_response(prompt):
    try:
        generated_text = nlp_model(prompt, max_length=200, min_length=50, do_sample=False)

        logging.debug(f"Raw model output: {generated_text}")

        if generated_text and isinstance(generated_text, list) and len(generated_text) > 0:
            response = generated_text[0]['summary_text']
            return clean_response(response)
        else:
            logging.warning("Unexpected output format from NLP model")
            return None
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return None

def clean_response(response):
    # Remove any leading/trailing whitespace
    cleaned = response.strip()
    return cleaned

# Route for the main chat page (index.html)
@app.route("/")
def index():
    return render_template('index.html')

# API route to handle chatbot interaction
@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get('query')
    response = rag_chatbot(user_query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

