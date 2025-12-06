import pandas as pd
import numpy as np
from src.services.embedding_service import embed_text
from src.services.text_cleaner import clean_text

async def process_request(cvs, jd):

    # 1) Read job description
    jd_txt = (await jd.read()).decode("utf-8", errors="ignore")
    jd_clean = clean_text(jd_txt)
    jd_emb = embed_text(jd_clean)

    results = []

    # 2) Process each CV file
    for f in cvs:
        cv_text = (await f.read()).decode("utf-8", errors="ignore")
        cv_clean = clean_text(cv_text)
        cv_emb = embed_text(cv_clean)

        # 3) Compute similarity
        score = float(
            np.dot(cv_emb, jd_emb) /
            (np.linalg.norm(cv_emb) * np.linalg.norm(jd_emb))
        )

        results.append({
            "cv_name": f.filename,
            "score": score
        })

    # 4) Sort by score descending
    df = pd.DataFrame(results).sort_values(by="score", ascending=False)

    return {
        "ranked_candidates": df.to_dict(orient="records")
    }
