import pandas as pd
from src.services.llm_service import build_scoring_prompt, call_ollama
from src.utils.json_utils import parse_llm_json
import json


def score_cvs(cv_df: pd.DataFrame, jd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a JSON list of:
      - all CVs (with cv_index, name, education, experience, certifications, skills, source="cv")
      - one JD (with matching fields, source="jd")
    Sends to LLaMA, gets back:
      [ {cv_index, name, certification_match_value, preferred_skills_match_value}, ... ]
    Returns as DataFrame.
    """

    cv_filtered = cv_df[
        ["cv_index", "name", "education", "experience", "certifications", "skills"]
    ].copy()
    cv_filtered["source"] = "cv"

    jd_filtered = jd_df[
        [
            "jd_index",
            "role_title",
            "core_responsibilities",
            "required_skills",
            "preferred_skills",
            "tools_and_technologies",
            "certification_requirements",
            "keywords_for_matching",
        ]
    ].copy()
    jd_filtered["source"] = "jd"

    merged = pd.concat([cv_filtered, jd_filtered], ignore_index=True)
    merged_json = merged.to_json(orient="records")

    prompt = build_scoring_prompt(merged_json)
    raw = call_ollama(prompt)
    parsed = parse_llm_json(raw)

    scoring_df = pd.DataFrame(parsed)
    return scoring_df
