# src/prediction_pipeline.py

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_models():
    # Load Sentence-Transformers model
    sbert_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

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
