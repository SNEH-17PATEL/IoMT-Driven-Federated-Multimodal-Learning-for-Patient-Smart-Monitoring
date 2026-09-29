"""Heuristic screening for unsupported clinical content in generated reports."""

import re
from decimal import Decimal, InvalidOperation


_UNSAFE_TERMS = {
    "medication or treatment recommendation": re.compile(
        r"\b(?:medications?|drugs?|treatment|therap(?:y|ies)|vasopressors?|"
        r"norepinephrine|noradrenaline|epinephrine|dopamine|vasopressin|"
        r"antibiotics?|antimicrobials?|insulin|heparin|steroids?|intubat(?:e|ion|ing)|"
        r"dialysis|crrt|surgery|procedures?|fluid bolus|crystalloids?)\b",
        re.IGNORECASE,
    ),
    "diagnosis-like term": re.compile(
        r"\b(?:diagnos(?:is|e|ed)|sepsis|septic shock|septicemia|pneumonia|"
        r"ards|acute respiratory distress syndrome|acute kidney injury|aki|"
        r"renal failure|myocardial infarction|heart attack|stroke|pulmonary embolism)\b",
        re.IGNORECASE,
    ),
    "dose language": re.compile(
        r"\b(?:dose|dosage|administer|prescribe|start|initiate|increase|decrease|"
        r"hold|stop|discontinue)\b|\b\d+(?:\.\d+)?\s*"
        r"(?:mg|mcg|µg|ug|g|ml|mL|units?|mcg/kg|min|mg/kg)\b",
        re.IGNORECASE,
    ),
}

_LABELED_MEASUREMENT = re.compile(
    r"\b(?:sofa(?:\s+score)?|heart\s*rate|hr(?:_mean|_std)?|"
    r"respiratory\s*rate|rr(?:_mean)?|spo2(?:_mean|_min)?|oxygen\s*saturation|"
    r"temperature|temp(?:_mean)?|sbp(?:_mean)?|dbp(?:_mean)?|map(?:_mean)?|"
    r"gcs(?:\s+eye\s+opening)?|stress\s*score)\b[^\d+-]{0,24}"
    r"([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_UNIT_MEASUREMENT = re.compile(
    r"([+-]?\d+(?:\.\d+)?)\s*(?:mmhg|bpm|br/min|breaths?\s*(?:/|per)\s*min|"
    r"°\s?[cf]|celsius|fahrenheit|%|percent)(?![a-z])",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])")


def _normalized_numbers(text):
    values = set()
    for match in _NUMBER.finditer(text):
        try:
            values.add(Decimal(match.group()).normalize())
        except InvalidOperation:
            continue
    return values


def screen_responses(responses, supplied_input):
    """Return safety findings for generated text, without claiming completeness."""
    allowed_numbers = _normalized_numbers(supplied_input)
    findings = []

    for response_index, response in enumerate(responses, start=1):
        response_findings = []
        for label, pattern in _UNSAFE_TERMS.items():
            if pattern.search(response):
                response_findings.append(label)

        measurements = [
            match.group(1)
            for pattern in (_LABELED_MEASUREMENT, _UNIT_MEASUREMENT)
            for match in pattern.finditer(response)
        ]
        unsupported = []
        for value in measurements:
            try:
                normalized = Decimal(value).normalize()
            except InvalidOperation:
                continue
            if normalized not in allowed_numbers and value not in unsupported:
                unsupported.append(value)
        if unsupported:
            response_findings.append(
                "numeric measurement not found in supplied inputs: " + ", ".join(unsupported)
            )

        if response_findings:
            findings.append({"response": response_index, "reasons": response_findings})

    return {"safe": not findings, "findings": findings}