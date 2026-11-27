"""
Branch-coverage unit tests for backend/services/llm_service.py.

The existing test_llm_service_unit.py covers the cache key. This file covers the
rest of the logic that does not need a real LLM:
  • <think> tag stripping
  • section parsing
  • prompt building (alert vs. non-alert, hospital vs. home care)
  • each of the 9 reliability scoring helpers
  • the judge call (success, bad JSON, exception)
  • generate_report end to end with a FAKE Groq client

No network access and no GROQ_API_KEY are needed.
"""
import pytest

from backend.services import llm_service as llm


# ── Helpers / fixtures ────────────────────────────────────────────────────────

def _predict_result(sofa=3.0, alert=7.0, **vitals_override):
    vitals = {"HR": 80, "RR": 16, "SpO2": 97, "Temp": 37.0, "SBP": 120, "DBP": 80,
              "MAP": 90, "GCS_eye": 4, "stress": 1.0}
    vitals.update(vitals_override)
    return {
        "patient_id": 1,
        "sofa_score": sofa,
        "risk_level": "Low Risk",
        "conformal_q": 3.28,
        "clinical_note": "stable overnight",
        "alert_threshold": alert,
        "vitals": vitals,
        "shap_top5": [{"feature": "latest_HR", "impact": 0.5}],
        "_vitals_block": "HR: 80 bpm",
        "_cv_block": "GCS Eye Opening: 4\nStress Score: 1.0/10",
        "_risk_block": "- none",
        "_explanation_block": "- none",
        "trend_text": "stable",
    }


class _FakeGroq:
    """Stands in for groq.Groq. `replies` is a list of strings (or Exceptions)
    returned/raised on successive chat.completions.create calls."""
    replies: list = []
    calls: list = []

    def __init__(self, api_key=None):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        _FakeGroq.calls.append(kwargs)
        item = _FakeGroq.replies.pop(0)
        if isinstance(item, Exception):
            raise item

        class _Msg:  # minimal response shape
            content = item

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


@pytest.fixture
def fake_groq(monkeypatch):
    _FakeGroq.replies = []
    _FakeGroq.calls = []
    monkeypatch.setattr(llm, "Groq", _FakeGroq)
    monkeypatch.setattr(llm, "_REPORT_CACHE", {})
    monkeypatch.setattr(llm, "_INFLIGHT", {})
    return _FakeGroq


GOOD_REPORT = (
    "CURRENT CONDITION — The patient is stable with HR 80 bpm and SpO2 97%.\n"
    "PROBABLE CAUSE — Normal physiology; no acute process.\n"
    "RISK FORECAST — Low risk of deterioration, continue to monitor.\n"
    "IMMEDIATE ACTIONS — Routine monitoring and reassess in one hour."
)
JUDGE_JSON = ('{"factual_accuracy":5,"clinical_appropriateness":5,"urgency_calibration":5,'
              '"completeness":5,"unsupported_claims":[],"dangerous_recommendation_detected":false,'
              '"dangerous_details":""}')


# ── _strip_thinking ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_strip_thinking_removes_closed_block():
    assert llm._strip_thinking("<think>secret</think>Answer") == "Answer"


@pytest.mark.unit
def test_strip_thinking_drops_unclosed_block():
    assert llm._strip_thinking("Visible<think>ran out of tokens") == "Visible"


@pytest.mark.unit
def test_strip_thinking_drops_text_before_stray_closing_tag():
    assert llm._strip_thinking("leaked reasoning</think>Answer") == "Answer"


@pytest.mark.unit
def test_strip_thinking_plain_text_is_just_trimmed():
    assert llm._strip_thinking("  hello  ") == "hello"


# ── _parse_llm_sections ───────────────────────────────────────────────────────

@pytest.mark.unit
def test_parse_sections_finds_all_four():
    s = llm._parse_llm_sections(GOOD_REPORT)
    assert set(s) == {"CURRENT CONDITION", "PROBABLE CAUSE", "RISK FORECAST", "IMMEDIATE ACTIONS"}
    assert "stable" in s["CURRENT CONDITION"]
    assert s["IMMEDIATE ACTIONS"].startswith("Routine")


@pytest.mark.unit
def test_parse_sections_accepts_markdown_and_numbering():
    text = "## 1. **CURRENT CONDITION**\nok\n## 2. **PROBABLE CAUSE**\nbecause"
    s = llm._parse_llm_sections(text)
    assert s["CURRENT CONDITION"] == "ok"
    assert s["PROBABLE CAUSE"] == "because"


@pytest.mark.unit
def test_parse_sections_missing_headings_gives_empty_dict():
    assert llm._parse_llm_sections("no headings here") == {}


# ── _build_prompt ─────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_prompt_hospital_alert_leads_with_alert_text():
    prompt, system = llm._build_prompt(_predict_result(sofa=8.0), "hospital")
    assert "CLINICAL ALERT" in prompt
    assert "IMMEDIATE ACTIONS" in prompt
    assert system == llm._HOSPITAL_SYSTEM_MSG


@pytest.mark.unit
def test_prompt_hospital_no_alert_has_no_alert_text():
    prompt, _ = llm._build_prompt(_predict_result(sofa=2.0), "hospital")
    assert "CLINICAL ALERT" not in prompt


@pytest.mark.unit
def test_prompt_homecare_alert_uses_patient_alert_text():
    prompt, system = llm._build_prompt(_predict_result(sofa=9.0), "homecare")
    assert "PATIENT ALERT" in prompt
    assert system == llm._HOMECARE_SYSTEM_MSG


@pytest.mark.unit
def test_prompt_homecare_no_alert():
    prompt, _ = llm._build_prompt(_predict_result(sofa=2.0), "homecare")
    assert "PATIENT ALERT" not in prompt


@pytest.mark.unit
def test_prompt_alert_threshold_is_inclusive():
    prompt, _ = llm._build_prompt(_predict_result(sofa=7.0, alert=7.0), "hospital")
    assert "CLINICAL ALERT" in prompt


# ── 1. factual_grounding_score ────────────────────────────────────────────────

@pytest.mark.unit
def test_grounding_all_normal_vitals_scores_one():
    assert llm.factual_grounding_score("anything", {"SpO2": 98, "MAP": 90}, 2.0) == 1.0


@pytest.mark.unit
def test_grounding_low_spo2_needs_oxygen_mention():
    v = {"SpO2": 85}
    assert llm.factual_grounding_score("give oxygen now", v, 2.0) == 1.0
    assert llm.factual_grounding_score("nothing relevant", v, 2.0) == 0.0


@pytest.mark.unit
def test_grounding_low_map_needs_pressure_mention():
    v = {"SpO2": 98, "MAP": 55}
    assert llm.factual_grounding_score("start vasopressor", v, 2.0) == 1.0
    assert llm.factual_grounding_score("all fine", v, 2.0) == 0.0


@pytest.mark.unit
def test_grounding_high_hr_needs_heart_mention():
    v = {"HR": 130}
    assert llm.factual_grounding_score("tachycardia noted", v, 2.0) == 1.0
    assert llm.factual_grounding_score("all fine", v, 2.0) == 0.0


@pytest.mark.unit
def test_grounding_high_rr_needs_breathing_mention():
    v = {"RR": 30}
    assert llm.factual_grounding_score("tachypnea present", v, 2.0) == 1.0
    assert llm.factual_grounding_score("all fine", v, 2.0) == 0.0


@pytest.mark.unit
def test_grounding_high_sofa_needs_urgency_language():
    assert llm.factual_grounding_score("this is critical", {}, 12.0) == 1.0
    assert llm.factual_grounding_score("all fine", {}, 12.0) == 0.0


@pytest.mark.unit
def test_grounding_moderate_sofa_needs_moderate_language():
    assert llm.factual_grounding_score("needs monitoring and attention", {}, 6.0) == 1.0
    assert llm.factual_grounding_score("zzz", {}, 6.0) == 0.0


# ── 2. llm_clinical_rules_score ───────────────────────────────────────────────

@pytest.mark.unit
def test_rules_none_fired_is_perfect():
    assert llm.llm_clinical_rules_score("x", {"SpO2": 98, "MAP": 90}, 2.0) == (1.0, [])


@pytest.mark.unit
def test_rules_violation_is_reported():
    score, viols = llm.llm_clinical_rules_score("nothing", {"SpO2": 80}, 2.0)
    assert score == 0.0
    assert any("oxygen" in v for v in viols)


@pytest.mark.unit
def test_rules_partial_score():
    # two rules fire (SpO2<90 and SOFA>=10); only one is satisfied
    score, viols = llm.llm_clinical_rules_score("give oxygen", {"SpO2": 80}, 12.0)
    assert score == pytest.approx(0.5)
    assert len(viols) == 1


@pytest.mark.unit
def test_rules_high_stress_and_gcs1_rules():
    v = {"stress": 7.0, "GCS_eye": 1}
    score, viols = llm.llm_clinical_rules_score("nothing", v, 2.0)
    assert score == 0.0 and len(viols) == 2
    score, viols = llm.llm_clinical_rules_score("pain, consciousness", v, 2.0)
    assert score == 1.0 and viols == []


# ── 3. shap_coherence_score ───────────────────────────────────────────────────

@pytest.mark.unit
def test_shap_empty_list_is_neutral():
    assert llm.shap_coherence_score("x", []) == 0.5


@pytest.mark.unit
def test_shap_tiny_impacts_are_ignored():
    assert llm.shap_coherence_score("x", [{"feature": "latest_HR", "impact": 0.01}]) == 0.5


@pytest.mark.unit
def test_shap_mentions_counted():
    top = [{"feature": "latest_HR", "impact": 0.5}, {"feature": "latest_MAP", "impact": -0.4}]
    assert llm.shap_coherence_score("tachycardia only", top) == pytest.approx(0.5)
    assert llm.shap_coherence_score("tachycardia and hypotension", top) == 1.0


@pytest.mark.unit
def test_shap_unknown_feature_matches_its_own_name():
    assert llm.shap_coherence_score("the foo bar value", [{"feature": "foo_bar", "impact": 1}]) == 1.0


# ── 4. response_structure_score ───────────────────────────────────────────────

@pytest.mark.unit
def test_structure_all_sections_and_length_is_one():
    text = GOOD_REPORT + " " + "word " * 250
    assert llm.response_structure_score(text) == pytest.approx(1.0)


@pytest.mark.unit
def test_structure_empty_is_zero():
    assert llm.response_structure_score("") == 0.0


# ── 5. severity_calibration_score ─────────────────────────────────────────────

@pytest.mark.unit
def test_severity_high_sofa():
    assert llm.severity_calibration_score("this is critical", 12.0) == 1.0
    assert llm.severity_calibration_score("all calm", 12.0) == 0.2


@pytest.mark.unit
def test_severity_moderate_sofa_always_ok():
    assert llm.severity_calibration_score("whatever", 6.0) == 1.0


@pytest.mark.unit
def test_severity_low_sofa_penalises_alarmist_text():
    assert llm.severity_calibration_score("urgent and life-threatening", 2.0) == 0.4
    assert llm.severity_calibration_score("urgent but stable, monitor", 2.0) == 1.0
    assert llm.severity_calibration_score("all calm", 2.0) == 1.0


# ── 6. contraindication_score ─────────────────────────────────────────────────

@pytest.mark.unit
def test_contra_clean_response():
    assert llm.contraindication_score("give fluids", {"MAP": 90}) == (1.0, [])


@pytest.mark.unit
def test_contra_nsaid_always_flagged():
    score, flags = llm.contraindication_score("take ibuprofen", {})
    assert score == 0.0 and "NSAIDs" in flags[0]


@pytest.mark.unit
def test_contra_beta_blocker_in_shock():
    score, flags = llm.contraindication_score("start metoprolol", {"MAP": 55, "HR": 120})
    assert score == 0.0 and "Beta-blockers" in flags[0]


@pytest.mark.unit
def test_contra_opioid_in_hypoxia():
    score, flags = llm.contraindication_score("give morphine", {"SpO2": 85})
    assert score == 0.0 and "opioids" in flags[0]


# ── 7. numeric_hallucination_score ────────────────────────────────────────────

@pytest.mark.unit
def test_numeric_no_numbers_is_perfect():
    assert llm.numeric_hallucination_score("no digits at all", {}, 3.0) == (1.0, [])


@pytest.mark.unit
def test_numeric_matching_value_is_allowed():
    score, bad = llm.numeric_hallucination_score("heart rate 80 bpm", {"HR": 80}, 3.0)
    assert score == 1.0 and bad == []


@pytest.mark.unit
def test_numeric_invented_value_is_flagged():
    score, bad = llm.numeric_hallucination_score("heart rate 187", {"HR": 80}, 3.0)
    assert bad == ["187"]
    assert score == pytest.approx(0.75)


@pytest.mark.unit
def test_numeric_repeated_value_counted_once():
    score, bad = llm.numeric_hallucination_score("HR 187 and heart rate 187", {"HR": 80}, 3.0)
    assert bad == ["187"]


@pytest.mark.unit
def test_numeric_unparseable_number_is_ignored(monkeypatch):
    # Force the Decimal conversion to fail so the InvalidOperation branch runs.
    import decimal

    class _Boom:
        def __init__(self, *_):
            raise decimal.InvalidOperation()

    monkeypatch.setattr(llm, "Decimal", _Boom)
    score, bad = llm.numeric_hallucination_score("heart rate 5", {"HR": 80}, 3.0)
    assert score == 1.0 and bad == []


@pytest.mark.unit
def test_numeric_standard_reference_value_is_allowed():
    # 65 mmHg is a clinical reference threshold, not a patient measurement
    score, bad = llm.numeric_hallucination_score("keep MAP above 65 mmHg", {"MAP": 90}, 3.0)
    assert bad == []


# ── call_judge_llm ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_judge_returns_parsed_json(fake_groq):
    fake_groq.replies = ["Here you go: " + JUDGE_JSON]
    out = llm.call_judge_llm("key", GOOD_REPORT, {"SpO2": 97}, 3.0, timeout=5)
    assert out["factual_accuracy"] == 5


@pytest.mark.unit
def test_judge_swallows_exceptions_and_returns_none(fake_groq):
    fake_groq.replies = [RuntimeError("network down")]
    assert llm.call_judge_llm("key", GOOD_REPORT, {}, 3.0, timeout=5) is None


@pytest.mark.unit
def test_judge_without_json_returns_none(fake_groq):
    fake_groq.replies = ["I cannot grade this"]
    assert llm.call_judge_llm("key", GOOD_REPORT, {}, 3.0, timeout=5) is None


@pytest.mark.unit
def test_judge_prompt_only_includes_first_900_chars(fake_groq):
    fake_groq.replies = [JUDGE_JSON]
    llm.call_judge_llm("key", "A" * 900 + "TAILMARKER", {}, 3.0, timeout=5)
    sent = fake_groq.calls[0]["messages"][0]["content"]
    assert "TAILMARKER" not in sent
    assert fake_groq.calls[0]["temperature"] == 0.0


# ── compute_reliability ───────────────────────────────────────────────────────

_V = {"HR": 80, "RR": 16, "SpO2": 97, "Temp": 37.0, "SBP": 120, "DBP": 80,
      "MAP": 90, "GCS_eye": 4, "stress": 1.0}


@pytest.mark.unit
def test_reliability_without_judge_uses_neutral_defaults():
    score, viols = llm.compute_reliability(GOOD_REPORT, _V, 3.0, [], None)
    assert 0.0 <= score <= 1.0
    assert viols == []


@pytest.mark.unit
def test_reliability_perfect_judge_scores_higher_than_no_judge():
    good = {"factual_accuracy": 5, "clinical_appropriateness": 5, "urgency_calibration": 5,
            "completeness": 5, "unsupported_claims": []}
    with_judge, _ = llm.compute_reliability(GOOD_REPORT, _V, 3.0, [], good)
    without, _ = llm.compute_reliability(GOOD_REPORT, _V, 3.0, [], None)
    assert with_judge > without


@pytest.mark.unit
def test_reliability_unsupported_claims_reduce_score_and_are_listed():
    judge = {"factual_accuracy": 3, "clinical_appropriateness": 3, "urgency_calibration": 3,
             "completeness": 3, "unsupported_claims": ["a", "b", "c"]}
    score, viols = llm.compute_reliability(GOOD_REPORT, _V, 3.0, [], judge)
    assert "Unsupported claim: a" in viols and "Unsupported claim: b" in viols
    assert "Unsupported claim: c" not in viols  # only the first two are shown


@pytest.mark.unit
def test_reliability_dangerous_recommendation_zeroes_contra_and_adds_flag():
    judge = {"factual_accuracy": 5, "clinical_appropriateness": 5, "urgency_calibration": 5,
             "completeness": 5, "unsupported_claims": [],
             "dangerous_recommendation_detected": True, "dangerous_details": "bad dose"}
    _, viols = llm.compute_reliability(GOOD_REPORT, _V, 3.0, [], judge)
    assert "AI Judge: bad dose" in viols


@pytest.mark.unit
def test_reliability_dangerous_without_details_adds_no_flag():
    judge = {"factual_accuracy": 5, "clinical_appropriateness": 5, "urgency_calibration": 5,
             "completeness": 5, "unsupported_claims": [],
             "dangerous_recommendation_detected": True}
    _, viols = llm.compute_reliability(GOOD_REPORT, _V, 3.0, [], judge)
    assert not any(v.startswith("AI Judge") for v in viols)


@pytest.mark.unit
def test_reliability_lists_hallucinated_numbers():
    _, viols = llm.compute_reliability("heart rate 187 bpm", _V, 3.0, [], None)
    assert "Hallucinated number in response: 187" in viols


# ── _reliability_label ────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("score,label", [
    (0.90, "High"), (0.75, "High"), (0.74, "Moderate"), (0.55, "Moderate"),
    (0.54, "Low"), (0.0, "Low"),
])
def test_reliability_label_thresholds(score, label):
    assert llm._reliability_label(score)[0] == label


# ── generate_report (fake Groq) ───────────────────────────────────────────────

@pytest.mark.unit
def test_generate_report_success_path(fake_groq):
    fake_groq.replies = [GOOD_REPORT, JUDGE_JSON]
    out = llm.generate_report(_predict_result(), "hospital", "key")
    assert out["error"] is None
    assert out["judge_used"] is True
    assert out["reliability_label"] in {"High", "Moderate", "Low"}
    assert set(out["sections"]) >= {"CURRENT CONDITION", "PROBABLE CAUSE",
                                    "RISK FORECAST", "IMMEDIATE ACTIONS"}
    assert out["care_mode"] == "hospital"
    # writer then judge
    assert fake_groq.calls[0]["temperature"] == 0.15
    assert fake_groq.calls[1]["temperature"] == 0.0


@pytest.mark.unit
def test_generate_report_is_cached_for_identical_input(fake_groq):
    fake_groq.replies = [GOOD_REPORT, JUDGE_JSON]
    pr = _predict_result()
    first = llm.generate_report(pr, "hospital", "key")
    second = llm.generate_report(pr, "hospital", "key")  # would pop from empty list if not cached
    assert second is first
    assert len(fake_groq.calls) == 2


@pytest.mark.unit
def test_generate_report_llm_failure_returns_error_and_is_not_cached(fake_groq):
    fake_groq.replies = [RuntimeError("groq down")]
    out = llm.generate_report(_predict_result(), "hospital", "key")
    assert out["sections"] is None
    assert out["reliability_score"] is None
    assert "unavailable" in out["error"]
    assert llm._REPORT_CACHE == {}


@pytest.mark.unit
def test_generate_report_empty_llm_text_uses_fallback_message(fake_groq):
    fake_groq.replies = ["<think>only reasoning</think>"]
    out = llm.generate_report(_predict_result(), "hospital", "key")
    assert out["sections"] is None
    assert out["raw_response"] == llm._FALLBACK_MSG


@pytest.mark.unit
def test_generate_report_judge_failure_falls_back_to_neutral(fake_groq):
    fake_groq.replies = [GOOD_REPORT, RuntimeError("judge down")]
    out = llm.generate_report(_predict_result(), "hospital", "key")
    assert out["judge_used"] is False
    assert out["error"] is None


@pytest.mark.unit
def test_generate_report_fills_missing_sections_with_empty_strings(fake_groq):
    fake_groq.replies = ["CURRENT CONDITION — only this one section.", JUDGE_JSON]
    out = llm.generate_report(_predict_result(), "homecare", "key")
    assert out["sections"]["PROBABLE CAUSE"] == ""
    assert out["sections"]["RISK FORECAST"] == ""
    assert out["sections"]["IMMEDIATE ACTIONS"] == ""


@pytest.mark.unit
def test_generate_report_low_reliability_sets_caution(fake_groq):
    bad_judge = ('{"factual_accuracy":1,"clinical_appropriateness":1,"urgency_calibration":1,'
                 '"completeness":1,"unsupported_claims":["x","y","z"],'
                 '"dangerous_recommendation_detected":true,"dangerous_details":"d"}')
    fake_groq.replies = ["ibuprofen. heart rate 187 bpm", bad_judge]
    out = llm.generate_report(_predict_result(), "hospital", "key")
    assert out["caution"] is True
    assert out["reliability_label"] == "Low"


@pytest.mark.unit
def test_generate_report_without_api_key_skips_judge(fake_groq):
    fake_groq.replies = [GOOD_REPORT]
    out = llm.generate_report(_predict_result(), "hospital", "")
    assert out["judge_used"] is False
    assert len(fake_groq.calls) == 1
