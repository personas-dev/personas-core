from jobmatch_gnn.data.dataset import CandidateRecord, DatasetBundle, InteractionRecord, JobRecord
from jobmatch_gnn.inference.service import HRKGRecommender, RecommendationRequest


def make_bundle() -> DatasetBundle:
    candidate = CandidateRecord(
        user_id="u1",
        text="python pytorch 推荐系统",
        skills=("python", "pytorch", "推荐"),
        city_ids=("530",),
        desired_city_ids=("530",),
        industries=("互联网",),
        desired_job_types=("算法工程师",),
        degree="本科",
        years_experience=5.0,
    )
    job = JobRecord(
        job_id="j1",
        title="算法工程师",
        text="python pytorch 推荐系统",
        skills=("python", "pytorch", "推荐"),
        city_id="530",
        job_type="算法工程师",
        min_degree="本科",
        min_years=3.0,
    )
    return DatasetBundle(
        candidates={"u1": candidate},
        jobs={"j1": job},
        interactions=[InteractionRecord("u1", "j1", 1.0, 0)],
        train_by_user={"u1": {"j1"}},
        valid_by_user={},
        test_by_user={"u1": {"j1"}},
        positive_by_user={"u1": {"j1"}},
        kg_stats={},
        stats={},
    )


def test_recommendation_response_contains_explanation() -> None:
    recommender = HRKGRecommender(make_bundle())
    response = recommender.recommend(RecommendationRequest(candidate_id="u1", model="rule", top_k=1))
    assert response.recommendations[0].job_id == "j1"
    assert "python" in response.recommendations[0].matched_skills
    assert response.recommendations[0].feature_evidence["city_match"] == 1.0


def test_spc_hgt_fallback_reports_actual_model() -> None:
    recommender = HRKGRecommender(make_bundle())
    response = recommender.recommend(RecommendationRequest(candidate_id="u1", model="spc_hgt", top_k=1))
    assert response.model == "rule"
    assert response.warnings
