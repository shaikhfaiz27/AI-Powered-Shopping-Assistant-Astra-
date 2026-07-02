import pandas as pd
import numpy as np
import faiss
import pickle

from sentence_transformers import SentenceTransformer

FILE = "ecommerce_data.xlsx"

products = pd.read_excel(
    FILE,
    sheet_name="Products"
)

products["search_text"] = (
    products["product_name"].astype(str)
    + " "
    + products["brand"].astype(str)
    + " "
    + products["category"].astype(str)
    + " "
    + products["description"].astype(str)
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(
    products["search_text"].tolist(),
    show_progress_bar=True
)

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(
    np.array(embeddings)
)

faiss.write_index(
    index,
    "products.index"
)

pickle.dump(
    products,
    open("products.pkl","wb")
)

print("Index Created")