import pandas as pd
from src.utils.file_utils import extract_text_from_file
from src.services.llm_service import build_jd_prompt, call_llm
from src.utils.json_utils import parse_llm_json
from src.services.embedding_service import clean_text, embed_text


def parse_jd_file(jd_path: str):
    """
    Takes JD file path (PDF/DOCX),
    returns a 1-row DataFrame with structured JD + embedding.
    """
    raw_text = extract_text_from_file(jd_path)
    prompt = build_jd_prompt(raw_text)
    raw = call_llm(prompt)
    parsed = parse_llm_json(raw)

    df = pd.DataFrame([parsed])

    df["combined_jd"] = (
            df["role_title"].astype(str) + " " +
            df["industry"].astype(str) + " " +
            df["core_responsibilities"].astype(str) + " " +
            df["required_skills"].astype(str) + " " +
            df["preferred_skills"].astype(str) + " " +
            df["tools_and_technologies"].astype(str) + " " +
            df["experience_years_required"].astype(str) + " " +
            df["education_requirements"].astype(str) + " " +
            df["keywords_for_matching"].astype(str)
    )

    df["clean_combined_jd"] = df["combined_jd"].apply(clean_text)
    df["jd_embedding"] = df["clean_combined_jd"].apply(embed_text)

    df = df.reset_index().rename(columns={"index": "jd_index"})
    return df
