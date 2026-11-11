
import glob
import os

from faster_whisper import WhisperModel

# Automatically select the latest extracted meeting audio
files = glob.glob(
    "Meeting_Recordings/uploaded_meeting_*.wav"
)

if not files:
    raise FileNotFoundError(
        "No uploaded meeting audio found!"
    )

audio_path = max(files, key=os.path.getmtime)

print("Testing audio:", audio_path)
print("\nLoading Faster-Whisper Medium model...")

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

print("Transcribing Tamil-English meeting...")

segments, info = model.transcribe(
    audio_path,
    language=None,
    task="transcribe",
    beam_size=5,
    vad_filter=True,
    condition_on_previous_text=True,
    initial_prompt=(
        "Tamil-English code-switching conversation "
        "about a college committee meeting. "
        "Wi-Fi issue, third floor signal strength, "
        "student feedback, computer lab, motherboard "
        "replacement, drinking water purifier, "
        "filter replacement, bus route, principal "
        "approval, classroom cleaning, afternoon "
        "inspection, timetable, elective class, "
        "complaints and action items. "
        "வணக்கம். நம்ம meeting-ல discuss பண்ண "
        "complaints-க்கு status update பண்ணப்போறோம். "
        "Use Tamil script for Tamil words and English "
        "script for English words. Do not translate."
    )
)

print("\nDetected language:", info.language)
print("\n========== TRANSCRIPT ==========\n")

for segment in segments:
    print(
        f"[{segment.start:.2f}s - "
        f"{segment.end:.2f}s] "
        f"{segment.text.strip()}"
    )

print("\n========== TEST COMPLETED ==========")
