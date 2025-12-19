from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes.process_routes import router as process_router

app = FastAPI(
    title="CV–JD Matching Engine",
    version="1.0.0",
    description="Parses CVs & JD with LLaMA3, embeds with MiniLM, matches via FAISS, and returns ranked candidates."
)

# CORS – relax for now, restrict later in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.get("/health")
def health():
    return {"status": "ok"}

# Main processing route
app.include_router(process_router, prefix="/api")
