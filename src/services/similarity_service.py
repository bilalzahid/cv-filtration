import numpy as np
import pandas as pd
import faiss


def compute_similarity(cv_df: pd.DataFrame, jd_df: pd.DataFrame, top_k: int = None):
    """
    Uses FAISS (cosine similarity via inner product on normalized vectors)
    to match CV embeddings against JD embedding.
    """

    cv_embeddings_df = cv_df[["cv_index", "name", "embedding"]].copy()
    jd_embeddings_df = jd_df[["jd_index", "role_title", "jd_embedding"]].copy()

    cv_matrix = np.vstack(cv_embeddings_df["embedding"].values)
    jd_matrix = np.vstack(jd_embeddings_df["jd_embedding"].values)

    def normalize(vecs):
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    cv_matrix_norm = normalize(cv_matrix)
    jd_matrix_norm = normalize(jd_matrix)

    dim = cv_matrix_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(cv_matrix_norm.astype("float32"))

    if top_k is None:
        top_k = cv_matrix_norm.shape[0]

    distances, indices = index.search(jd_matrix_norm.astype("float32"), top_k)

    results = []
    for jd_row, (cv_idx_list, sim_list) in enumerate(zip(indices, distances)):
        jd_entry = jd_embeddings_df.iloc[jd_row]
        jd_id = jd_entry["jd_index"]
        jd_title = jd_entry["role_title"]

        for rank, (cv_id, score) in enumerate(zip(cv_idx_list, sim_list)):
            cv_entry = cv_embeddings_df.iloc[cv_id]
            results.append(
                {
                    "jd_index": jd_id,
                    "job_title": jd_title,
                    "cv_index": cv_entry["cv_index"],
                    "cv_name": cv_entry["name"],
                    "similarity": float(score),
                    "rank_similarity": rank + 1,
                }
            )

    matches_df = pd.DataFrame(results)
    return matches_df
