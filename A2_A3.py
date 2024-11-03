#Q2 and Q3,RAG Application with UI(form.html too)

import cohere
from bs4 import BeautifulSoup
import requests
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import aiohttp
import asyncio
import time

#Initialize Cohere client
co = cohere.Client("zaYlVqyiv7uu1xwCXUEZqtcocz8qnMo7pI3Px2Oh")


#Scrape data from a URL
def scrape_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()


#Chunk the data into smaller pieces
def chunk_data(text, chunk_size=200):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


#Store chunks in a Faiss index
def store_chunks_in_faiss(chunks):
    vectors = np.random.rand(len(chunks), 512).astype('float32')
    index = faiss.IndexFlatL2(512)
    index.add(vectors)
    return index, vectors


#Retrieve relevant chunks based on a query
def retrieve_relevant_chunks(query, index, chunks):
    query_vector = np.random.rand(1, 512).astype('float32')
    distances, indices = index.search(query_vector, 3)
    return [chunks[i] for i in indices[0]]


#Generate text based on the provided prompt and context
async def generate_text(prompt, context, temp=0.5, num_responses=3):
    responses = []
    for _ in range(num_responses):
        response = co.chat_stream(
            message=prompt,
            model="command-r",
            temperature=float(temp),
            preamble=context
        )
        generated_text = ""
        for event in response:
            if event.event_type == "text-generation":
                generated_text += event.text
        responses.append(generated_text)
    return responses


#LLM for comparison between statements
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b"
api_token = "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"
headers = {"Authorization": f"Bearer {api_token}"}


async def query(payload):
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=payload) as response:
            return await response.json()

'''
async def compare_using_api(context, sentences):
    data = await query({
        "inputs": {
            "source_sentence": context,
            "sentences": sentences
        }
    })

    if not isinstance(data, list):
        raise ValueError("Expected a list of scores, but got: {}".format(data))

    max_score = max(data)
    max_index = data.index(max_score)
    return sentences[max_index]
'''
async def compare_using_api(context, sentences, retries=5, delay=20):
    for _ in range(retries):
        data = await query({
            "inputs": {
                "source_sentence": context,
                "sentences": sentences
            }
        })

        if isinstance(data, list):
            max_score = max(data)
            max_index = data.index(max_score)
            return sentences[max_index]
        else:
            print(f"Model loading. Retrying in {delay} seconds...")
            time.sleep(delay)

    raise ValueError("Expected a list of scores, but got: {}".format(data))


app = Flask(__name__)
CORS(app)

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('form.html')


@app.route('/ask', methods=['POST'])
async def ask_question():
    try:
        data = request.json
        question = data['question']
        chunks = retrieve_relevant_chunks(question, index, stored_chunks)
        context = " ".join(chunks)
        answers = await generate_text(question, context)
        solution = await compare_using_api(question, answers)

        return jsonify(answers=answers, selected_answer=solution)
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    raw_data = scrape_data("https://en.wikipedia.org/wiki/Luke_Skywalker")
    stored_chunks = chunk_data(raw_data)
    index, vectors = store_chunks_in_faiss(stored_chunks)
    app.run(debug=True)
