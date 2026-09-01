
import voyageai
from config import VOYAGE_API_KEY
vo = voyageai.Client(api_key=VOYAGE_API_KEY)  # reads VOYAGE_API_KEY 

MODEL_NAME = "voyage-4-large"


def embed_documents(texts):
    
    response = vo.embed(
        texts=texts,
        model=MODEL_NAME,
        input_type="document",
    )
    return response.embeddings


def embed_query(text):
   
    response = vo.embed(
        texts=[text],
        model=MODEL_NAME,
        input_type="query",
    )
    return response.embeddings[0]


if __name__ == "__main__":
    # Quick manual test
    doc_vectors = embed_documents(["The sky is blue.", "Python is a programming language."])
    print(f"Embedded {len(doc_vectors)} documents, each with {len(doc_vectors[0])} dimensions")

    query_vector = embed_query("What color is the sky?")
    print(f"Embedded query, {len(query_vector)} dimensions")