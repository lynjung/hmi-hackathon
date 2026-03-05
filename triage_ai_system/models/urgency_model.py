"""Rule-based urgency scoring model for triage."""

from __future__ import annotations

from dataclasses import dataclass


SYMPTOM_WEIGHTS = {
    "chest pain": 5,
    "shortness of breath": 5,
    "dizziness": 3,
    "fever": 2,
    "vomiting": 2,
    "headache": 1,
    "fatigue": 1,
    "cough": 1,
}

HISTORY_WEIGHTS = {
    "hypertension": 2,
    "diabetes": 2,
    "heart disease": 4,
    "asthma": 2,
}

FAMILY_WEIGHTS = {
    "heart disease": 3,
    "stroke": 3,
    "diabetes": 2,
}


@dataclass
class UrgencyResult:
    score: int
    urgency_level: str
    explanation: str


class UrgencyModel:
    """Computes urgency score and human-readable rationale."""

    def predict(self, symptoms: list[str], medical_history: list[str], family_history: list[str], age: int) -> UrgencyResult:
        symptom_score = sum(SYMPTOM_WEIGHTS.get(item, 0) for item in symptoms)
        history_score = sum(HISTORY_WEIGHTS.get(item, 0) for item in medical_history)
        family_score = sum(FAMILY_WEIGHTS.get(item, 0) for item in family_history)
        age_score = 2 if age > 60 else 0

        total_score = symptom_score + history_score + family_score + age_score
        urgency_level = self._classify(total_score)

        explanation_parts = [
            f"Symptom severity contribution: {symptom_score}",
            f"Medical history risk contribution: {history_score}",
            f"Family history risk contribution: {family_score}",
            f"Age contribution: {age_score}",
        ]

        if symptoms:
            explanation_parts.append(f"Detected symptoms: {', '.join(symptoms)}")
        if medical_history:
            explanation_parts.append(f"Medical history risks: {', '.join(medical_history)}")
        if family_history:
            explanation_parts.append(f"Family history risks: {', '.join(family_history)}")

        explanation = " | ".join(explanation_parts)

        return UrgencyResult(score=total_score, urgency_level=urgency_level, explanation=explanation)

    @staticmethod
    def _classify(score: int) -> str:
        if score >= 9:
            return "Emergency"
        if score >= 6:
            return "Urgent"
        if score >= 3:
            return "Moderate"
        return "Low"
