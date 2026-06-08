"""Optional FastAPI adapter for the HR KG-GNN recommender."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from jobmatch_gnn.inference.service import CandidateProfileInput, HRKGRecommender, RecommendationError, RecommendationRequest


def create_app(config_path: str = "configs/train_gnn.yaml", run_dir: str = "experiments/runs/baseline_sample_gpu") -> Any:
    """Create a FastAPI app for frontend HTTP calls.

    FastAPI is an optional dependency. Install with:
    `uv pip install --python .venv/bin/python 'fastapi>=0.115' 'uvicorn[standard]>=0.30'`.
    """

    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FastAPI adapter requires fastapi and pydantic. Install the API optional dependencies first.") from exc

    class CandidateProfilePayload(BaseModel):  # type: ignore[no-redef]
        candidate_id: str = "anonymous"
        skills: list[str] = Field(default_factory=list)
        experience_text: str = ""
        desired_city_ids: list[str] = Field(default_factory=list)
        current_city_ids: list[str] = Field(default_factory=list)
        industries: list[str] = Field(default_factory=list)
        desired_job_types: list[str] = Field(default_factory=list)
        degree: str = ""
        years_experience: float = 0.0

    class RecommendationPayload(BaseModel):  # type: ignore[no-redef]
        candidate_id: str | None = None
        candidate_profile: CandidateProfilePayload | None = None
        model: str = "spc_hgt"
        top_k: int = 10
        include_explanations: bool = True
        request_id: str | None = None

    recommender = HRKGRecommender.from_config(Path(config_path), Path(run_dir), model_name="spc_hgt")
    app = FastAPI(title="HR KG-GNN Recommendation API", version="v1")

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "candidate_count": len(recommender.bundle.candidates), "job_count": len(recommender.bundle.jobs)}

    @app.post("/api/v1/recommendations")
    def recommend(payload: RecommendationPayload) -> dict[str, Any]:
        candidate_profile = None
        if payload.candidate_profile is not None:
            candidate_profile = CandidateProfileInput(**payload.candidate_profile.model_dump())
        request = RecommendationRequest(
            candidate_id=payload.candidate_id,
            candidate_profile=candidate_profile,
            model=payload.model,
            top_k=payload.top_k,
            include_explanations=payload.include_explanations,
            request_id=payload.request_id,
        )
        try:
            return asdict(recommender.recommend(request))
        except RecommendationError as exc:
            raise HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message}) from exc

    return app


app = create_app()
