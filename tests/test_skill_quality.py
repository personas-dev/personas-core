from jobmatch_gnn.data.preprocess_v2 import build_automaton, build_skill_vocab, match_job_skills
from jobmatch_gnn.data.skill_quality import SkillQualityGate, SkillReference


def test_generic_terms_and_short_english_are_filtered_from_vocab():
	gate = SkillQualityGate.from_config({"enabled": True})
	rows = [
		{"experience": "Python|自我评价|沟通|or|C++|客户"},
		{"experience": "Python|学历|or|C++|责任心"},
	]

	vocab = build_skill_vocab(rows, min_count=1, quality=gate)

	assert "python" in vocab
	assert "c++" in vocab
	assert "沟通" in vocab
	assert "责任心" in vocab
	assert "自我评价" not in vocab
	assert "客户" not in vocab
	assert "学历" not in vocab
	assert "or" not in vocab


def test_ascii_boundary_blocks_embedded_job_text_matches():
	gate = SkillQualityGate.from_config({"enabled": True})
	vocab = {"go": 0, "sql": 1}
	automaton = build_automaton(vocab)

	matches = match_job_skills(
		automaton,
		title="golang开发工程师",
		description="需要 sql 数据处理经验, 也需要 go 服务开发。",
		max_skills=10,
		title_weight=2.0,
		quality=gate,
	)

	assert {idx for idx, _ in matches} == {0, 1}


def test_osta_occupation_reference_terms_are_not_skill_nodes():
	reference = SkillReference(
		occupations={"计算机程序设计员"},
		categories={"工程技术人员"},
		work_types={"低代码开发员"},
	)
	gate = SkillQualityGate(reference=reference)

	assert not gate.decide("计算机程序设计员").keep
	assert not gate.decide("工程技术人员").keep
	assert not gate.decide("低代码开发员").keep
	assert gate.decide("python").keep
