# Frontend API Protocol: HR KG-GNN Recommendation

Date: 2026-06-08
API version: `v1`
Owner: personas-core algorithm module

## Purpose

This document defines the JSON protocol that the frontend can call through the backend recommendation service. The API wraps the HR KG-GNN matching algorithm and returns Top-K job recommendations with structured explanation evidence.

The current implementation exposes a pure Python service in `jobmatch_gnn.inference.service` and an optional FastAPI adapter in `jobmatch_gnn.inference.api`.

## Runtime

Recommended local runtime:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend cu130 -e . pytest ruff 'torch>=2.12'
```

Optional HTTP server dependencies:

```bash
uv pip install --python .venv/bin/python 'fastapi>=0.115' 'uvicorn[standard]>=0.30'
```

Start the API server:

```bash
.venv/bin/uvicorn jobmatch_gnn.inference.api:app --host 0.0.0.0 --port 8000
```

The default API app loads:

- Config: `configs/train_gnn.yaml`
- Run directory: `experiments/runs/baseline_sample_gpu`
- Preferred trained prediction order: `predictions/spc_hgt.json` when available

## Base URL

Local development:

```text
http://localhost:8000/api/v1
```

## Health Check

```http
GET /api/v1/health
```

Response:

```json
{
  "status": "ok",
  "candidate_count": 784,
  "job_count": 3000
}
```

## Recommend Jobs

```http
POST /api/v1/recommendations
Content-Type: application/json
```

### Request Body

Use either `candidate_id` for an existing loaded candidate or `candidate_profile` for an ad-hoc candidate profile.

```json
{
  "request_id": "web-20260608-0001",
  "candidate_id": "17e1b9f107dd1214bd78dec6d91593a4",
  "model": "spc_hgt",
  "top_k": 10,
  "include_explanations": true
}
```

Ad-hoc profile request:

```json
{
  "request_id": "web-20260608-0002",
  "model": "rule",
  "top_k": 5,
  "include_explanations": true,
  "candidate_profile": {
    "candidate_id": "frontend-demo-user",
    "skills": ["Python", "PyTorch", "推荐系统"],
    "experience_text": "5年推荐算法经验，熟悉 Python、PyTorch、召回排序和特征工程",
    "desired_city_ids": ["530"],
    "current_city_ids": ["530"],
    "industries": ["互联网"],
    "desired_job_types": ["算法工程师"],
    "degree": "本科",
    "years_experience": 5
  }
}
```

### Request Fields

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `request_id` | string | no | generated UUID | Frontend trace id. Echoed in response. |
| `candidate_id` | string | conditional | null | Existing candidate id loaded from `data/datasets.zip`. Required when `candidate_profile` is absent. |
| `candidate_profile` | object | conditional | null | Ad-hoc candidate profile. Required when `candidate_id` is absent. |
| `model` | string | no | `spc_hgt` | One of `spc_hgt`, `rule`, `bm25`, `semantic_hash`. |
| `top_k` | integer | no | 10 | Number of jobs to return. Server clamps to `[1, 100]`. |
| `include_explanations` | boolean | no | true | Whether to return matched skills, missing skills, paths, and reasons. |

### Candidate Profile Fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `candidate_id` | string | no | Frontend/user id for tracing. |
| `skills` | string[] | no | Explicit normalized or raw skill tags. |
| `experience_text` | string | no | Resume/profile text used for token extraction. |
| `desired_city_ids` | string[] | no | Expected job city ids. |
| `current_city_ids` | string[] | no | Current candidate city ids. |
| `industries` | string[] | no | Preferred/current industries. |
| `desired_job_types` | string[] | no | Preferred job type labels. |
| `degree` | string | no | Degree label such as `大专`, `本科`, `硕士`. |
| `years_experience` | number | no | Approximate years of work experience. |

## Response Body

```json
{
  "request_id": "web-20260608-0001",
  "model": "spc_hgt",
  "candidate_id": "17e1b9f107dd1214bd78dec6d91593a4",
  "generated_at": "2026-06-08T08:30:00.000000+00:00",
  "top_k": 10,
  "warnings": [],
  "recommendations": [
    {
      "rank": 1,
      "job_id": "4ce99de185f55bea127ccd74c4bbf0ad",
      "title": "土建工程造价员",
      "score": 1.0,
      "city_id": "551",
      "job_type": "工程造价/预结算",
      "matched_skills": ["工程", "预算", "软件"],
      "missing_skills": ["cad", "广联达"],
      "graph_paths": [
        "Candidate:17e1b9f107dd1214bd78dec6d91593a4 -> Skill:工程 <- Job:4ce99de185f55bea127ccd74c4bbf0ad"
      ],
      "reasons": ["技能覆盖度 0.42", "城市匹配", "学历满足岗位要求"],
      "feature_evidence": {
        "required_skill_coverage": 0.42,
        "missing_skill_ratio": 0.58,
        "city_match": 1.0,
        "education_match": 1.0,
        "experience_match": 1.0,
        "job_type_match": 1.0,
        "skill_path_count_log1p": 2.08
      }
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|---|---|---|
| `request_id` | string | Frontend trace id or generated UUID. |
| `model` | string | Model actually used by the service. |
| `candidate_id` | string | Candidate id used for ranking. |
| `generated_at` | string | UTC ISO-8601 timestamp. |
| `top_k` | integer | Number requested after server-side clamp. |
| `warnings` | string[] | Non-fatal fallback messages. |
| `recommendations` | object[] | Ranked job list. |

### Recommendation Fields

| Field | Type | Description |
|---|---|---|
| `rank` | integer | 1-based rank. |
| `job_id` | string | Job id from source table `jd_no`. |
| `title` | string | Job title. |
| `score` | number | Model-specific ranking score. For trained prediction order, this is rank confidence `1/rank`. |
| `city_id` | string | Job city id. |
| `job_type` | string | Job type/subtype label. |
| `matched_skills` | string[] | Shared candidate/job skill tokens. |
| `missing_skills` | string[] | Job skill tokens absent from the candidate profile. |
| `graph_paths` | string[] | Evidence paths in `Candidate -> Skill <- Job` form. |
| `reasons` | string[] | Human-readable reason snippets for UI display. |
| `feature_evidence` | object | Numeric evidence for explanation chips or debug panels. |

## Error Response

FastAPI returns standard HTTP errors. Algorithm validation errors use this detail shape:

```json
{
  "detail": {
    "code": "CANDIDATE_NOT_FOUND",
    "message": "Candidate 'missing-user' was not found in the loaded dataset sample."
  }
}
```

| Code | HTTP | Meaning | Suggested Frontend Handling |
|---|---:|---|---|
| `MISSING_CANDIDATE` | 400 | Neither `candidate_id` nor `candidate_profile` was provided. | Prompt user to select or fill a candidate profile. |
| `CANDIDATE_NOT_FOUND` | 400 | Candidate id is not in the loaded backend sample. | Retry with `candidate_profile` or refresh candidate list. |
| `UNSUPPORTED_MODEL` | 400 | Unknown model string. | Fall back to `spc_hgt` or `rule`. |
| `MODEL_ARTIFACT_MISSING` | 200 + warning | Trained prediction order is unavailable. | Display warning badge if needed; service falls back to Rule. |

## Frontend Display Contract

Recommended card layout fields:

- Main title: `title`
- Subtitle: `job_type`, `city_id`
- Score badge: `score`
- Explanation chips: first 3 items from `reasons`
- Matched skills: `matched_skills`
- Skill gaps: `missing_skills`
- Debug drawer: `feature_evidence` and `graph_paths`

The frontend should not assume `score` is calibrated across different `model` values. Use rank order as the primary display signal.

## JavaScript Example

```ts
const response = await fetch('/api/v1/recommendations', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    request_id: crypto.randomUUID(),
    candidate_id: selectedCandidateId,
    model: 'spc_hgt',
    top_k: 10,
    include_explanations: true,
  }),
});

if (!response.ok) {
  const error = await response.json();
  throw new Error(error.detail?.message ?? 'Recommendation request failed');
}

const data = await response.json();
renderRecommendationCards(data.recommendations);
```

## Python Backend Example

```python
from pathlib import Path
from jobmatch_gnn.inference import HRKGRecommender, RecommendationRequest

service = HRKGRecommender.from_config(
    Path('configs/train_gnn.yaml'),
    Path('experiments/runs/baseline_sample_gpu'),
)
response = service.recommend(RecommendationRequest(candidate_id='17e1b9f107dd1214bd78dec6d91593a4', top_k=10))
payload = service.to_dict(response)
```

## Compatibility Notes

- `spc_hgt` uses saved training prediction order when `experiments/runs/baseline_sample_gpu/predictions/spc_hgt.json` exists.
- If trained order is missing or the candidate was not in the training sample, `spc_hgt` falls back to Rule ranking and returns a warning.
- Ad-hoc `candidate_profile` requests currently use feature/rule-compatible ranking unless a future online encoder is added.
- `semantic_hash` is a deterministic fallback and should be replaced by real SBERT embeddings before product claims about semantic matching.
