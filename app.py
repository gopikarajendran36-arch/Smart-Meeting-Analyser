
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
import soundfile as sf
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Committee Meeting Advisor",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Committee Meeting Advisor")

st.write(
    "Upload a meeting video to generate a transcript, "
    "identify speakers, create a summary, and extract possible action items."
)


# --------------------------------------------------
# LOAD AI MODELS ONLY WHEN REQUIRED
# --------------------------------------------------
@st.cache_resource
def load_whisper():
    from faster_whisper import WhisperModel

    # Try GPU first. If CUDA is unavailable, use CPU.
    try:
        model = WhisperModel(
            "medium",
            device="cuda",
            compute_type="float16"
        )
        return model, "GPU"

    except Exception:
        model = WhisperModel(
            "medium",
            device="cpu",
            compute_type="int8"
        )
        return model, "CPU"


@st.cache_resource
def load_diarization():
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1"
    )

    pipeline.to(torch.device("cpu"))
    return pipeline


# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------
st.subheader("📁 Upload Meeting Video")

uploaded_file = st.file_uploader(
    "Choose a meeting video",
    type=["mp4", "mov", "avi", "mkv"],
    key="meeting_video_uploader"
)

if uploaded_file is None:
    st.info("Please upload an MP4, MOV, AVI, or MKV video to continue.")

else:
    video_extension = Path(uploaded_file.name).suffix or ".mp4"
    video_path = Path("uploaded_meeting" + video_extension)

    video_path.write_bytes(uploaded_file.getbuffer())

    st.success("Video uploaded successfully!")

    st.video(str(video_path))

    # --------------------------------------------------
    # GENERATE TRANSCRIPT
    # --------------------------------------------------
    if st.button("🎤 Generate Transcript with Speakers"):

        try:
            # --------------------------------------------------
            # EXTRACT AUDIO
            # --------------------------------------------------
            with st.spinner("Extracting audio from video..."):

                audio_path = "meeting_audio.wav"
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

                process = subprocess.run(
                    [
                        ffmpeg_path,
                        "-y",
                        "-i", str(video_path),
                        "-vn",
                        "-ac", "1",
                        "-ar", "16000",
                        "-sample_fmt", "s16",
                        audio_path
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )

            st.success("Audio extracted successfully!")

            # --------------------------------------------------
            # FASTER-WHISPER TRANSCRIPTION
            # --------------------------------------------------
            with st.spinner(
                "Transcribing meeting. This may take some time..."
            ):

                whisper_model, device_used = load_whisper()

                st.info(f"Transcription device: {device_used}")

                segments, info = whisper_model.transcribe(
                    audio_path,
                    task="transcribe",
                    temperature=0,
                    condition_on_previous_text=False,
                    vad_filter=False
                )

                transcript_segments = []

                for segment in segments:
                    transcript_segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    })

            if not transcript_segments:
                st.warning("No speech was detected in the uploaded video.")
                st.stop()

            # --------------------------------------------------
            # SPEAKER DIARIZATION
            # --------------------------------------------------
            with st.spinner("Identifying speakers..."):

                import torch

                audio, sample_rate = sf.read(
                    audio_path,
                    dtype="float32"
                )

                # Convert audio to channels x samples
                if audio.ndim == 1:
                    waveform = torch.from_numpy(audio).unsqueeze(0)
                else:
                    waveform = torch.from_numpy(audio.T)

                diarization_pipeline = load_diarization()

                diarization_result = diarization_pipeline({
                    "waveform": waveform,
                    "sample_rate": sample_rate
                })

                speaker_turns = [
                    (turn.start, turn.end, speaker)
                    for turn, speaker
                    in diarization_result.speaker_diarization
                ]

            # --------------------------------------------------
            # MATCH TRANSCRIPT SEGMENTS WITH SPEAKERS
            # --------------------------------------------------
            transcript_lines = []
            transcript_records = []

            for segment in transcript_segments:

                start = segment["start"]
                end = segment["end"]
                spoken_text = segment["text"]

                speaker_overlaps = {}

                for turn_start, turn_end, speaker in speaker_turns:

                    overlap = max(
                        0.0,
                        min(end, turn_end) - max(start, turn_start)
                    )

                    if overlap > 0:
                        speaker_overlaps[speaker] = (
                            speaker_overlaps.get(speaker, 0.0)
                            + overlap
                        )

                if speaker_overlaps:
                    speaker = max(
                        speaker_overlaps,
                        key=speaker_overlaps.get
                    )
                else:
                    speaker = "SPEAKER_UNKNOWN"

                transcript_records.append({
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": spoken_text
                })

                transcript_lines.append(
                    f"[{start:.2f}s - {end:.2f}s] "
                    f"{speaker}: {spoken_text}"
                )

            transcript = "\n".join(transcript_lines)

            # --------------------------------------------------
            # DISPLAY TRANSCRIPT
            # --------------------------------------------------
            st.success(
                "Transcription and speaker identification completed!"
            )

            st.subheader("📝 Meeting Transcript")

            st.text_area(
                "Transcript with speaker labels",
                transcript,
                height=350
            )

            st.download_button(
                label="📥 Download Transcript",
                data=transcript,
                file_name="meeting_transcript_with_speakers.txt",
                mime="text/plain"
            )

            # --------------------------------------------------
            # RAW SPEAKER TIMESTAMPS
            # --------------------------------------------------
            with st.expander("View raw speaker timestamps"):

                for start, end, speaker in speaker_turns:
                    st.write(
                        f"[{start:.2f}s - {end:.2f}s] {speaker}"
                    )

            # --------------------------------------------------
            # MEETING SUMMARY
            # --------------------------------------------------
            st.subheader("📊 Meeting Summary and Action Items")

            clean_sentences = [
                record["text"]
                for record in transcript_records
                if record["text"].strip()
            ]

            if clean_sentences:

                try:
                    from sklearn.feature_extraction.text import (
                        TfidfVectorizer
                    )

                    vectorizer = TfidfVectorizer(
                        stop_words="english"
                    )

                    matrix = vectorizer.fit_transform(clean_sentences)
                    scores = matrix.sum(axis=1).A1

                    ranked_indices = sorted(
                        range(len(clean_sentences)),
                        key=lambda i: scores[i],
                        reverse=True
                    )

                    selected_indices = sorted(
                        ranked_indices[:min(3, len(clean_sentences))]
                    )

                    summary = "\n".join(
                        "• " + clean_sentences[i]
                        for i in selected_indices
                    )

                except ValueError:
                    summary = "\n".join(
                        "• " + sentence
                        for sentence in clean_sentences[:3]
                    )

                st.markdown("**Key sentences (automatic summary)**")
                st.write(summary)

            else:
                summary = "No summary available."
                st.info(summary)

            # --------------------------------------------------
            # ACTION ITEM EXTRACTION
            # --------------------------------------------------
            action_items = []
            proposals = []

            commitment_pattern = re.compile(
                r"\bI\s+will\s+(.+)",
                re.IGNORECASE
            )

            request_pattern = re.compile(
                r"\b(?:please|need to|must|should)\s+(.+)",
                re.IGNORECASE
            )

            proposal_pattern = re.compile(
                r"\b(?:I suggest|suggest|propose|we could|we can)\s+(.+)",
                re.IGNORECASE
            )

            deadline_pattern = re.compile(
                r"\b(today|tomorrow|tonight|next week|"
                r"monday|tuesday|wednesday|thursday|"
                r"friday|saturday|sunday)\b",
                re.IGNORECASE
            )

            for record in transcript_records:

                sentence = record["text"].strip()
                speaker = record["speaker"]
                start = record["start"]
                end = record["end"]

                commitment = commitment_pattern.search(sentence)
                request = request_pattern.search(sentence)
                proposal = proposal_pattern.search(sentence)

                deadline_match = deadline_pattern.search(sentence)

                deadline = (
                    deadline_match.group(0).capitalize()
                    if deadline_match
                    else "Not specified"
                )

                if commitment:

                    action_items.append({
                        "task": commitment.group(1).strip().rstrip("."),
                        "speaker": speaker,
                        "start": start,
                        "end": end,
                        "deadline": deadline,
                        "type": "Explicit commitment",
                        "evidence": sentence
                    })

                elif request:

                    action_items.append({
                        "task": request.group(1).strip().rstrip("."),
                        "speaker": "Not confirmed",
                        "start": start,
                        "end": end,
                        "deadline": deadline,
                        "type": "Request or obligation",
                        "evidence": sentence
                    })

                elif proposal:

                    proposals.append({
                        "task": proposal.group(1).strip().rstrip("."),
                        "speaker": speaker,
                        "start": start,
                        "end": end,
                        "evidence": sentence
                    })

            # --------------------------------------------------
            # DISPLAY ACTION ITEMS
            # --------------------------------------------------
            st.markdown("**Confirmed commitments and possible tasks**")

            st.caption(
                "Automatically extracted candidates. "
                "Review them before treating them as assigned tasks."
            )

            if action_items:

                for index, item in enumerate(action_items, start=1):

                    with st.container(border=True):

                        st.markdown(
                            f"**Task {index}: {item['task']}**"
                        )

                        st.write(f"Type: {item['type']}")
                        st.write(
                            f"Responsible person: {item['speaker']}"
                        )
                        st.write(
                            f"Possible deadline: {item['deadline']}"
                        )
                        st.write(
                            f"Evidence timestamp: "
                            f"{item['start']:.2f}s–{item['end']:.2f}s"
                        )
                        st.write(
                            f"Original statement: {item['evidence']}"
                        )
                        st.write("Status: Needs confirmation")

            else:
                st.info(
                    "No explicit commitments or requests detected."
                )

            # --------------------------------------------------
            # DISPLAY PROPOSALS
            # --------------------------------------------------
            if proposals:

                st.markdown(
                    "**Proposals — not confirmed decisions**"
                )

                for item in proposals:
                    st.write(
                        f"• {item['task']} — proposed by "
                        f"{item['speaker']} "
                        f"({item['start']:.2f}s–{item['end']:.2f}s)"
                    )

            # --------------------------------------------------
            # DOWNLOAD MEETING ANALYSIS
            # --------------------------------------------------
            report_lines = [
                "MEETING SUMMARY",
                "",
                summary,
                "",
                "POSSIBLE ACTION ITEMS",
                ""
            ]

            for index, item in enumerate(action_items, start=1):

                report_lines.extend([
                    f"{index}. Task: {item['task']}",
                    f"   Type: {item['type']}",
                    f"   Responsible person: {item['speaker']}",
                    f"   Possible deadline: {item['deadline']}",
                    f"   Evidence: {item['evidence']}",
                    f"   Timestamp: {item['start']:.2f}s–{item['end']:.2f}s",
                    "   Status: Needs confirmation",
                    ""
                ])

            report_lines.extend(["PROPOSALS", ""])

            for item in proposals:
                report_lines.append(
                    f"- {item['task']} "
                    f"(proposed by {item['speaker']})"
                )

            report = "\n".join(report_lines)

            st.download_button(
                label="📥 Download Meeting Analysis",
                data=report,
                file_name="meeting_analysis.txt",
                mime="text/plain"
            )

        except Exception as error:
            st.error(f"Processing failed: {error}")
