from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import audit, claims, health, underwriting
from app.db import init_db

app = FastAPI(
    title="Multimodal Cross-Source Consistency Verification API (motor claims + commercial underwriting)",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],  # SvelteKit dev + preview
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(claims.router)
app.include_router(audit.router)
app.include_router(underwriting.router)


@app.on_event("startup")
async def on_startup() -> None:
    await init_db()
