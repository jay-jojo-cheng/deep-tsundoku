from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TITLE_EMBEDDING_DIRNAME = Path(__file__).resolve().parent / "artifacts"
TITLE_EMBEDDING_FILE = "title_matching_embeddings.npy"

DATA_DIRNAME = Path(__file__).resolve().parent.parent.parent / "data"
books_csv_file = "books_emb_8_tidy.csv"

class TextToAsin:
    """Serves as a bridge between phase 0 and phase 1 by converting detecting text into book ASIN.
    
    # example usage:
    phase_half = TextToAsin()
    query_strings = ["crime and punishment dostoevsky", "harry potter and the prisoner"]
    phase_half.title_to_asin(query_strings)
    """

    def __init__(self, title_emb_path=None, product_emb_path=None):
        self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
        if title_emb_path is None:
            title_emb_path = TITLE_EMBEDDING_DIRNAME / TITLE_EMBEDDING_FILE
        self.reference_embeddings = np.load(title_emb_path)

        if product_emb_path is None:
            title_emb_path = DATA_DIRNAME / books_csv_file
        self.df = pd.read_csv(title_emb_path).dropna()
    
    def return_k_similar(self, query_embeddings, k=1):
        top_k_titles = np.array([])
        for query in query_embeddings:
            similarity_mat = cosine_similarity(query, self.reference_embeddings)
            similarity_score = similarity_mat[0]
            if k == 1:
                top_k_titles = np.append(top_k_titles, np.argmax(similarity_score).reshape(1, -1))
            elif k is not None:
                top_k_titles = np.append(top_k_titles, np.flip(similarity_score.argsort()[-k:][::1]).reshape(1, -1))
        return top_k_titles
    
    # k is number of titles to extract
    def title_to_asin(self, query_strings, k=1) -> str:
        query_embeddings = [self.model.encode([s]) for s in query_strings]
        similar_item_idx = self.return_k_similar(query_embeddings, k)
        return self.df.iloc[similar_item_idx, ][['asin', 'title']]


