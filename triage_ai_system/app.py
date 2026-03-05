"""Streamlit MVP for an offline AI-powered triage assistant."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from models.symptom_nlp import SymptomExtractor
from models.urgency_model import UrgencyModel
from models.whisper_model import WhisperTranscriber
from queue.patient_queue import PatientPriorityQueue, PatientRecord
from utils.risk_scoring import urgency_color

APP_DIR = Path(__file__).parent
SYMPTOM_LIBRARY_PATH = APP_DIR / "data" / "symptom_library.json"

MEDICAL_HISTORY_OPTIONS = ["hypertension", "diabetes", "heart disease", "asthma"]
FAMILY_HISTORY_OPTIONS = ["heart disease", "stroke", "diabetes"]


@st.cache_resource
def load_whisper() -> WhisperTranscriber:
    return WhisperTranscriber(model_name="base")


@st.cache_resource
def load_symptom_extractor() -> SymptomExtractor:
    return SymptomExtractor(symptom_library_path=SYMPTOM_LIBRARY_PATH)


@st.cache_resource
def load_urgency_model() -> UrgencyModel:
    return UrgencyModel()


def setup_state() -> None:
    if "patient_queue" not in st.session_state:
        st.session_state.patient_queue = PatientPriorityQueue()
        st.session_state.patient_queue.seed_demo_patients()
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def render_level_badge(level: str) -> None:
    color = urgency_color(level)
    st.markdown(
        f"<div style='padding:0.5rem;border-radius:0.5rem;background:{color};color:#111;font-weight:700;text-align:center;'>{level.upper()}</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Offline Triage AI", page_icon="🩺", layout="wide")
    st.title("🩺 AI-Powered Offline Triage Assistant")
    st.caption("Local MVP for low-connectivity clinics using Whisper + BioBERT + rule-based triage")

    setup_state()

    try:
        whisper_model = load_whisper()
        symptom_extractor = load_symptom_extractor()
        urgency_model = load_urgency_model()
    except Exception as exc:
        st.error(
            "Model loading failed. Ensure Whisper and BioBERT models are downloaded locally before offline use."
        )
        st.exception(exc)
        st.stop()

    intake_tab, doctor_tab = st.tabs(["Patient Intake", "Doctor Dashboard"])

    with intake_tab:
        st.subheader("Step 1 — Patient Intake")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Patient Name", placeholder="John Doe")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            medical_history = st.multiselect("Medical History", MEDICAL_HISTORY_OPTIONS)
            family_history = st.multiselect("Family History", FAMILY_HISTORY_OPTIONS)

        with col2:
            st.markdown("**Symptoms Input**")
            input_mode = st.radio("Choose input type", ["Text", "Voice Upload"], horizontal=True)
            text_input = st.text_area("Symptom Description", placeholder="I feel dizzy and my chest hurts when I breathe.")
            audio_file = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])

        if st.button("Analyze Symptoms", type="primary"):
            if not name.strip():
                st.warning("Please provide patient name.")
                st.stop()

            transcript = text_input.strip()
            if input_mode == "Voice Upload":
                if audio_file is None:
                    st.warning("Please upload an audio file for voice mode.")
                    st.stop()
                transcript = whisper_model.transcribe_audio(audio_file.getvalue(), suffix=Path(audio_file.name).suffix)
                st.markdown("### Step 2 — Speech Processing")
                st.info(f"**Transcribed Input:** {transcript}")

            st.markdown("### Step 3 — NLP Symptom Extraction")
            detected_symptoms = symptom_extractor.extract_symptoms(transcript)
            if detected_symptoms:
                st.success("Detected Symptoms: " + ", ".join(detected_symptoms))
            else:
                st.warning("No symptoms detected confidently. Please provide more detail.")

            st.markdown("### Step 4 & 5 — Risk Integration and Urgency Scoring")
            urgency = urgency_model.predict(
                symptoms=detected_symptoms,
                medical_history=medical_history,
                family_history=family_history,
                age=int(age),
            )

            render_level_badge(urgency.urgency_level)
            st.metric("Urgency Score", urgency.score)
            st.write("**Explanation:**", urgency.explanation)

            patient = PatientRecord(
                name=name.strip(),
                age=int(age),
                symptoms=detected_symptoms,
                medical_history=medical_history,
                family_history=family_history,
                urgency_score=urgency.score,
                urgency_level=urgency.urgency_level,
                explanation=urgency.explanation,
                transcript=transcript,
                timestamp=time.time(),
            )
            st.session_state.patient_queue.add_patient(patient)
            st.session_state.last_result = patient
            st.success("Patient added to priority queue.")

    with doctor_tab:
        st.subheader("Step 6 & 7 — Queue Management + Doctor Dashboard")
        queue_items = st.session_state.patient_queue.get_queue()

        if not queue_items:
            st.info("No patients in queue yet.")
            return

        table_rows = []
        for idx, patient in enumerate(queue_items, start=1):
            table_rows.append(
                {
                    "Position": idx,
                    "Patient": patient.name,
                    "Symptoms": ", ".join(patient.symptoms) if patient.symptoms else "N/A",
                    "Urgency": patient.urgency_level,
                    "Score": patient.urgency_score,
                }
            )

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        selected_name = st.selectbox("Select patient for details", [p.name for p in queue_items])
        selected = next(p for p in queue_items if p.name == selected_name)

        st.markdown("### Triage Explanation")
        c1, c2 = st.columns([1, 2])
        with c1:
            render_level_badge(selected.urgency_level)
            st.metric("Score", selected.urgency_score)
        with c2:
            st.write(f"**Patient:** {selected.name}")
            st.write(f"**Age:** {selected.age}")
            st.write(f"**Transcript:** {selected.transcript}")
            st.write(f"**Symptoms:** {', '.join(selected.symptoms) if selected.symptoms else 'N/A'}")
            st.write(
                f"**Medical History:** {', '.join(selected.medical_history) if selected.medical_history else 'None'}"
            )
            st.write(
                f"**Family History:** {', '.join(selected.family_history) if selected.family_history else 'None'}"
            )
            st.write(f"**Reasoning:** {selected.explanation}")


if __name__ == "__main__":
    main()
