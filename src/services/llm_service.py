import requests

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, stream: bool = False) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }
    resp = requests.post(OLLAMA_GENERATE_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["response"]


# === CV PARSING PROMPT ===
EXTRACTION_PROMPT = """
You are an AI assistant tasked with extracting structured data from a CV or Resume.

Given the text of the CV, extract the following fields and return ONLY a valid JSON object:
{
  "name": string,
  "email": string or null,
  "phone": string or null,
  "education": [string],
  "experience": [string],
  "experience_duration": float or null,
  "certifications": [string],
  "skills": [string]
}

Instructions:
- For array fields use [] if nothing found.
- Deduplicate entries for skills, certifications, and experience.
- Do NOT invent data. Only use what appears in the CV.
- Certifications MUST be real certifications explicitly present in text.
- Experience duration: estimate total professional years (float) or null if impossible.

Respond with a single valid JSON object. No explanations.
"""

def build_cv_prompt(cv_text: str) -> str:
    return EXTRACTION_PROMPT + "\n\nCV TEXT:\n" + cv_text


# === JD PARSING PROMPT ===
JD_EXTRACTION_PROMPT = """
You are an AI Job Description Parser used in a Candidate Matching Engine.

Extract the following fields and return ONLY a valid JSON object:
{
  "role_title": string,
  "industry": string,
  "core_responsibilities": [string],
  "required_skills": [string],
  "preferred_skills": [string],
  "tools_and_technologies": [string],
  "experience_years_required": string,
  "education_requirements": [string],
  "certification_requirements": [string],
  "keywords_for_matching": [string]
}

Rules:
- Use short phrases.
- Do NOT invent fields; if nothing found, use empty string or [].
- This output will be used to match against candidate CVs.

Respond with a single valid JSON object. No explanations.
"""

def build_jd_prompt(jd_text: str) -> str:
    return JD_EXTRACTION_PROMPT + "\n\nJD TEXT:\n" + jd_text


# === SCORING PROMPT (CERT + PREFERRED SKILLS) ===
FLAGS_EXTRACTION_PROMPT = """
You are an expert CV-to-Job evaluator engine.

You will receive a JSON list that has:
- multiple CV records
- one job description record

Each CV record has fields like:
- cv_index, name, education, experience, certifications, skills, source="cv"

The JD record has fields like:
- jd_index, role_title, core_responsibilities, required_skills, preferred_skills,
  tools_and_technologies, certification_requirements, keywords_for_matching, source="jd"

Goal:
For each CV (source="cv"), output a score object based on:

1) certification_match_value
   → how well CV certifications match JD's certification_requirements
2) preferred_skills_match_value
   → how well CV skills overlap with JD preferred_skills

Scoring Rules (each between 0.1 and 0.5):
- 0.50 → very strong match
- 0.40 → good match
- 0.30 → moderate match
- 0.20 → weak match
- 0.10 → very weak or almost no match

Output format:
Return an array of JSON objects. Each object MUST be:
{
  "cv_index": <number>,
  "name": "<string>",
  "certification_match_value": <float between 0.1 and 0.5>,
  "preferred_skills_match_value": <float between 0.1 and 0.5>
}

Important:
- Returns raw JSON (no quotes-wrapped JSON, no explanations).
"""

def build_scoring_prompt(merged_json: str) -> str:
    return FLAGS_EXTRACTION_PROMPT + "\n\nINPUT JSON:\n" + merged_json
