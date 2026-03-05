"""Priority queue utilities for triage ordering."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass


@dataclass
class PatientRecord:
    name: str
    age: int
    symptoms: list[str]
    medical_history: list[str]
    family_history: list[str]
    urgency_score: int
    urgency_level: str
    explanation: str
    transcript: str
    timestamp: float


class PatientPriorityQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[tuple[int, float], PatientRecord]] = []

    def add_patient(self, patient: PatientRecord) -> None:
        priority_key = (-patient.urgency_score, patient.timestamp)
        heapq.heappush(self._heap, (priority_key, patient))

    def get_queue(self) -> list[PatientRecord]:
        return [item[1] for item in sorted(self._heap, key=lambda x: x[0])]

    def seed_demo_patients(self) -> None:
        if self._heap:
            return

        demo = [
            PatientRecord(
                name="Demo Patient A",
                age=65,
                symptoms=["chest pain", "dizziness"],
                medical_history=["hypertension"],
                family_history=["heart disease"],
                urgency_score=10,
                urgency_level="Emergency",
                explanation="Chest pain + dizziness with cardiovascular risk factors.",
                transcript="I have chest pain and dizziness.",
                timestamp=time.time(),
            ),
            PatientRecord(
                name="Demo Patient B",
                age=25,
                symptoms=["fever", "cough"],
                medical_history=[],
                family_history=[],
                urgency_score=3,
                urgency_level="Moderate",
                explanation="Mild infectious symptoms without major risk factors.",
                transcript="I have fever and cough.",
                timestamp=time.time() + 1,
            ),
        ]

        for p in demo:
            self.add_patient(p)
