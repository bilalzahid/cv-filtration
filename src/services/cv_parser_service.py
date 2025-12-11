import os
import pandas as pd
from src.utils.file_utils import extract_text_from_file
from src.services.llm_service import build_cv_prompt, call_ollama
from src.utils.json_utils import parse_llm_json
from src.services.embedding_service import clean_text, embed_text


def parse_cv_files(cv_paths):
    """
    Takes a list of CV file paths (PDF/DOCX),
    returns a DataFrame with:
    [cv_index, name, email, phone, education, experience,
     experience_duration, certifications, skills,
     file_name, profile_text, clean_profile_text, embedding]
    """
    all_parsed = []

    for path in cv_paths:
        raw_text = extract_text_from_file(path)
        prompt = build_cv_prompt(raw_text)
        raw = call_ollama(prompt)
        parsed = parse_llm_json(raw)
        parsed["file_name"] = os.path.basename(path)
        all_parsed.append(parsed)

    df = pd.DataFrame(all_parsed)

    df["profile_text"] = (
            df["experience"].astype(str) + " " +
            df["education"].astype(str) + " " +
            df["skills"].astype(str) + " " +
            df["certifications"].astype(str)
    )

    df["clean_profile_text"] = df["profile_text"].apply(clean_text)
    df["embedding"] = df["clean_profile_text"].apply(embed_text)

    df = df.reset_index().rename(columns={"index": "cv_index"})
    return df
