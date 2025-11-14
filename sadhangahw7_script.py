import requests
from bs4 import BeautifulSoup
import numpy as np
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def download_and_embed():
    """Download documents, chunk, and compute embeddings"""
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]

    documents = []
    for url in urls:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')

        for tag in soup.find_all(['script', 'style', 'sup']):
            tag.decompose()

        content = soup.find('div', {'id': 'mw-content-text'})
        if content:
            paragraphs = content.find_all('p', recursive=True)
            text = ' '.join([p.get_text().strip() for p in paragraphs[:10] if p.get_text().strip()])
            if text:
                documents.append(text)

    print(f"Downloaded {len(documents)} documents")

    # Semantic chunking
    def chunk_text(text, max_words=100):
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            words = sentence.split()
            if current_length + len(words) > max_words and current_chunk:
                chunks.append(' '.join(current_chunk).strip())
                current_chunk = []
                current_length = 0
            current_chunk.extend(words)
            current_length += len(words)

        if current_chunk:
            chunks.append(' '.join(current_chunk).strip())

        return [c for c in chunks if c]

    chunks = []
    for doc in documents:
        chunks.extend(chunk_text(doc))

    print(f"Created {len(chunks)} chunks")

    # Compute embeddings
    embeddings = []
    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)

    embeddings_array = np.array(embeddings)
    np.savez('embeddings.npz', embeddings=embeddings_array, chunks=chunks)
    print("Embeddings saved\n")

    return embeddings_array, chunks

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_context(query, embeddings, chunks, top_k=3):
    """Retrieve most relevant chunks for a query"""
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = np.array(query_response.data[0].embedding)

    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [chunks[i] for i in top_indices]

# Main execution
if os.path.exists('embeddings.npz'):
    print("Loading cached embeddings...")
    data = np.load('embeddings.npz', allow_pickle=True)
    embeddings = data['embeddings']
    chunks = data['chunks'].tolist()
    print(f"Loaded {len(chunks)} chunks\n")
else:
    print("No cache found. Downloading and computing embeddings...")
    embeddings, chunks = download_and_embed()

# RAG query
query = "What is machine learning and how does it relate to artificial intelligence?"
print(f"Query: {query}\n")

relevant_chunks = retrieve_context(query, embeddings, chunks)
print(f"Retrieved {len(relevant_chunks)} relevant chunks\n")

# Build context
context = "\n\n".join(relevant_chunks)

# Send to chat API
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer based on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
)

print("Response:")
print(response.choices[0].message.content)
