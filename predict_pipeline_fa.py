import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

def load_models():
    # Load Sentence-Transformers model
    sbert_model = SentenceTransformer("sbert_model")

    # Load FAISS index
    index = faiss.read_index("faiss_index.index")

    # Load T5 model and tokenizer
    text_generation_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    text_generation_model = T5ForConditionalGeneration.from_pretrained("t5-base")

    # Load text chunks
    text_chunks = np.load("text_chunks.npy", allow_pickle=True)
    
    return sbert_model, index, text_generation_tokenizer, text_generation_model, text_chunks

def generate_embeddings(sbert_model, texts):
    return sbert_model.encode(texts)

def generate_long_answer(text_generation_tokenizer, text_generation_model, question, context, max_length=300):
    input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = text_generation_tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=1024)
    
    # Generate the output
    outputs = text_generation_model.generate(
        inputs,
        max_length=max_length,
        num_beams=5,
        early_stopping=True
    )
    answer = text_generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Load models at the start of the application
sbert_model, faiss_index, text_generation_tokenizer, text_generation_model, text_chunks = load_models()

@app.route('/')
def index():
    return render_template('fa.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data['question']

    # Generate embedding for the question
    question_embedding = generate_embeddings(sbert_model, [question])

    # Search for the most similar chunks
    k = 5  # Number of most similar chunks to retrieve
    distances, indices = faiss_index.search(question_embedding, k)

    # Retrieve the most relevant chunks
    relevant_chunks = [text_chunks[i] for i in indices[0]]

    # Concatenate relevant chunks into a single context
    context = "\n\n".join(relevant_chunks)

    # Generate the long answer
    answer = generate_long_answer(text_generation_tokenizer, text_generation_model, question, context)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
