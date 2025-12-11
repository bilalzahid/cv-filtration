import uuid
from typing import List
from fastapi import UploadFile

from src.utils.file_utils import create_session_folder, save_uploaded_file, cleanup_session
from src.services.cv_parser_service import parse_cv_files
from src.services.jd_parser_service import parse_jd_file
from src.services.similarity_service import compute_similarity
from src.services.scoring_service import score_cvs


async def run_complete_pipeline(jd_file: UploadFile, cv_files: List[UploadFile]):
    """
    Orchestrates:
    - Save files to /tmp/session_uuid
    - Parse CVs & JD via LLaMA
    - Embed texts via MiniLM
    - Compute similarity via FAISS
    - Score with certification & preferred skills weights
    - Return final ranked JSON
    """
    session_id = f"session_{uuid.uuid4().hex}"
    session_path = create_session_folder(session_id)

    try:
        # 1) Save uploaded files
        jd_path = save_uploaded_file(jd_file, session_path)
        cv_paths = [save_uploaded_file(cv, session_path) for cv in cv_files]

        # 2) Parse CVs
        cv_df = parse_cv_files(cv_paths)

        # 3) Parse JD
        jd_df = parse_jd_file(jd_path)

        # 4) Compute similarity
        matches_df = compute_similarity(cv_df, jd_df)

        # 5) Score using LLM (cert + preferred skills)
        scoring_df = score_cvs(cv_df, jd_df)

        # 6) Merge similarity + scoring
        final_df = matches_df.merge(scoring_df, on="cv_index", how="inner")

        final_df["final_score"] = final_df["similarity"] * (
                final_df["certification_match_value"]
                + final_df["preferred_skills_match_value"]
        )

        # 7) Attach file_name, skills, JD metadata
        cv_meta = cv_df[["cv_index", "file_name", "skills"]]
        jd_meta = jd_df[
            ["jd_index", "role_title", "experience_years_required",
             "required_skills", "preferred_skills"]
        ]

        final_df = final_df.merge(cv_meta, on="cv_index", how="left")
        final_df = final_df.merge(jd_meta, on="jd_index", how="left")

        final_df = final_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        final_df["rank"] = final_df.index + 1

        # 8) Build response JSON
        jd_row = jd_df.iloc[0]

        response = {
            "batch_id": f"BATCH{uuid.uuid4().hex[:8]}",
            "status": "completed",
            "jd_summary": {
                "title": jd_row.get("role_title"),
                "experience_required": jd_row.get("experience_years_required"),
                "required_skills": jd_row.get("required_skills", []),
                "preferred_skills": jd_row.get("preferred_skills", []),
            },
            "results": [],
            "errors": [],
        }

        for _, row in final_df.iterrows():
            response["results"].append(
                {
                    "file_name": row.get("file_name"),
                    "candidate_name": row.get("cv_name"),
                    "skills": row.get("skills", []),
                    "overall_score": float(row.get("final_score", 0.0)),
                    "rank": int(row.get("rank", 0)),
                }
            )

        return response

    finally:
        cleanup_session(session_path)
