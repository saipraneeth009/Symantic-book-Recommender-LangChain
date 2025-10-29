import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import os

# Prefer Mistral if the API key is available; otherwise fall back to a local HF embedding
MISTRAL_KEY = os.getenv("MISTRALAI_API_KEY")
if MISTRAL_KEY:
    try:
        from langchain_mistralai import MistralAIEmbeddings
        embedding = MistralAIEmbeddings(model="mistral-embed")
    except Exception:
        # If Mistral import fails despite key being present, clear key so fallback runs below
        embedding = None
        MISTRAL_KEY = None
else:
    embedding = None

import gradio as gr

load_dotenv()

books = pd.read_csv("data/books_with_emotions.csv")

'''
as book cover thumbnails are taken from the links provided in dataset and each book is not of same resolution,
using the below part to get every thumbnail to same size
'''
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800" 
# books that don't have cover uses this image
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "app/cover-not-available.jpg",
    books["large_thumbnail"],
)

#from vector_search notebook
raw_doc = TextLoader("data/tagged_description.txt",encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_doc)
if embedding is None:
    # Try langchain's HuggingFaceEmbeddings first (uses sentence-transformers models)
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Using local HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)")
    except Exception:
        # Final fallback: use sentence-transformers directly and wrap it to match the expected API
        try:
            from sentence_transformers import SentenceTransformer

            class LocalEmbedding:
                def __init__(self, model_name="all-MiniLM-L6-v2"):
                    # model name in HF hub for sentence-transformers mappings
                    self.model = SentenceTransformer(model_name)

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    # returns list of vectors
                    return [list(map(float, vec)) for vec in self.model.encode(texts, show_progress_bar=True)]

                def embed_query(self, text: str) -> list[float]:
                    return list(map(float, self.model.encode([text])[0]))

            embedding = LocalEmbedding("all-MiniLM-L6-v2")
            print("Using sentence-transformers LocalEmbedding (all-MiniLM-L6-v2)")
        except Exception:
            raise RuntimeError(
                "No embedding backend available. Either set MISTRALAI_API_KEY or install 'sentence-transformers' (pip install sentence-transformers)"
            )

db_books = Chroma.from_documents(documents, embedding=embedding)

def retrieve_semantic_recom(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:
    
    
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)
    
    if category != "ALL":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)
        
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprise":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspense":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="Sadness", ascending=False, inplace=True)
        
    return book_recs 

def recom_books(
    query: str,
    category: str,
    tone: str,

):
    
    recoms = retrieve_semantic_recom(query, category, tone)
    results = []
    
    for _, row in recoms.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_desc = " ".join(truncated_desc_split[:50]) + "..." 
        
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
            
        captions = f"{row['title']} by {authors_str}: {truncated_desc}"
        results.append((row["large_thumbnail"], captions))
        
    return results  

categories = ["ALL"] + sorted(books["simple_categories"].unique())
tones = ["ALL"] + ["Happy", "Surprise", "Angry", "Suspense", "Sad"]

with gr.Blocks(theme = gr.themes.Ocean()) as dashboard:
    gr.Markdown("# Semantic Book Recommender System")
    
    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter description of the book you are looking for:",
            placeholder="e.g., A story about a ......",)
        category = gr.Dropdown( choices=categories, label="Select category:", value="ALL")
        tone = gr.Dropdown( choices=tones, label="Select emotional tone:", value="ALL")
        submit_btn = gr.Button("Get Recommendations")
        
    gr.Markdown("## Recommended Books")
    output = gr.Gallery(label= "Recommended Books", columns= 8, rows= 2)
    
    submit_btn.click(
        fn = recom_books,
        inputs = [user_query, category, tone],
        outputs = output,
    )

if __name__ == "__main__":
    dashboard.launch()






