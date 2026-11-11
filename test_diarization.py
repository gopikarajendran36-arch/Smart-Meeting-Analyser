
import os
import glob
import soundfile as sf
import torch
import numpy as np

from pyannote.audio import Pipeline

# STEP 1: Find latest uploaded meeting audio

audio_files = glob.glob(
    "Meeting_Recordings/uploaded_meeting_*.wav"
)

if not audio_files:
    raise FileNotFoundError(
        "No uploaded meeting audio found!"
    )

AUDIO_FILE = max(audio_files, key=os.path.getmtime)

print("Audio file:", AUDIO_FILE)

# STEP 2: Load audio

audio, sample_rate = sf.read(
    AUDIO_FILE,
    dtype="float32"
)

# Convert stereo to mono if needed
if audio.ndim == 2:
    audio = np.mean(audio, axis=1)

waveform = torch.from_numpy(audio).unsqueeze(0)

# STEP 3: Load pyannote model

print("\nLoading speaker diarization model...")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1"
)

pipeline.to(torch.device("cpu"))

print("Model loaded successfully!")

# STEP 4: Identify speakers

print("\nStarting speaker diarization...")

output = pipeline(
    {
        "waveform": waveform,
        "sample_rate": sample_rate
    },
    min_speakers=2,
    max_speakers=2
)

# STEP 5: Display results

print("\n========== SPEAKER SUMMARY ==========")

speaker_durations = {}

for turn, speaker in output.speaker_diarization:
    duration = turn.end - turn.start

    speaker_durations[speaker] = (
        speaker_durations.get(speaker, 0) + duration
    )

for speaker, duration in speaker_durations.items():
    print(f"{speaker}: {duration:.2f} seconds")

print("\nDiarization completed successfully!")

# STEP 6: Save individual speaker audio

speaker_audio_dir = "Meeting_Recordings/speaker_audio"
os.makedirs(speaker_audio_dir, exist_ok=True)

for turn, speaker in output.speaker_diarization:
    start = int(turn.start * sample_rate)
    end = int(turn.end * sample_rate)

    speaker_audio = audio[start:end]

    if len(speaker_audio) == 0:
        continue

    output_file = os.path.join(
        speaker_audio_dir,
        f"{speaker}_{turn.start:.2f}.wav"
    )

    sf.write(
        output_file,
        speaker_audio,
        sample_rate
    )

print("\nIndividual speaker audio saved!")
print("Folder:", speaker_audio_dir)

