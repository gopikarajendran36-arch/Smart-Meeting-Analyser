
import sys
import os
import time
import threading
import subprocess
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundcard as sc
import soundfile as sf
import imageio_ffmpeg

from faster_whisper import WhisperModel
import torch
from pyannote.audio import Pipeline
from PySide6.QtGui import QTextCursor

from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout,
    QMessageBox, QFrame, QTabWidget, QFileDialog
)


SAMPLE_RATE = 16000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "Meeting_Recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)


# ==================================================
# LIVE AUDIO RECORDING WORKER
# ==================================================

class AudioRecordingWorker(QThread):

    status_changed = Signal(str)
    recording_saved = Signal(str, str)
    error_occurred = Signal(str)

    def __init__(self, mic_device_id=1):
        super().__init__()

        self.mic_device_id = mic_device_id
        self.stop_event = threading.Event()

        self.mic_chunks = []
        self.system_chunks = []

        self.data_lock = threading.Lock()
        self.errors = []

    def stop_recording(self):
        self.stop_event.set()

    def record_microphone(self):
        try:
            def callback(indata, frames, time_info, status):
                if status:
                    self.status_changed.emit(
                        f"Microphone notice: {status}"
                    )

                if not self.stop_event.is_set():
                    with self.data_lock:
                        self.mic_chunks.append(indata.copy())

            with sd.InputStream(
                device=self.mic_device_id,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=callback,
                blocksize=1600
            ):
                while not self.stop_event.wait(0.1):
                    pass

        except Exception as exc:
            self.errors.append(f"Microphone: {exc}")
            self.stop_event.set()

    def record_system_audio(self):
        try:
            speaker = sc.default_speaker()

            loopback = sc.get_microphone(
                id=str(speaker.name),
                include_loopback=True
            )

            with loopback.recorder(
                samplerate=SAMPLE_RATE
            ) as recorder:

                while not self.stop_event.is_set():

                    audio = recorder.record(
                        numframes=1600
                    )

                    if audio is not None and len(audio) > 0:
                        audio = np.asarray(
                            audio, dtype=np.float32
                        )

                        # Convert stereo to mono for storage.
                        if audio.ndim == 2:
                            audio = np.mean(
                                audio, axis=1, keepdims=True
                            )

                        with self.data_lock:
                            self.system_chunks.append(audio)

        except Exception as exc:
            self.errors.append(f"System audio: {exc}")
            self.stop_event.set()

    def run(self):

        self.status_changed.emit("Starting audio devices...")

        mic_thread = threading.Thread(
            target=self.record_microphone,
            daemon=True
        )

        system_thread = threading.Thread(
            target=self.record_system_audio,
            daemon=True
        )

        mic_thread.start()
        system_thread.start()

        self.status_changed.emit(
            "Recording microphone and system audio..."
        )

        while not self.stop_event.wait(0.1):
            pass

        mic_thread.join(timeout=3)
        system_thread.join(timeout=3)

        if self.errors:
            self.error_occurred.emit(
                "\n".join(self.errors)
            )
            return

        with self.data_lock:
            mic_chunks = list(self.mic_chunks)
            system_chunks = list(self.system_chunks)

        if not mic_chunks or not system_chunks:
            self.error_occurred.emit(
                "No audio was captured from one or both sources."
            )
            return

        try:
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            mic_path = os.path.join(
                RECORDINGS_DIR,
                f"meeting_mic_{timestamp}.wav"
            )

            system_path = os.path.join(
                RECORDINGS_DIR,
                f"meeting_system_{timestamp}.wav"
            )

            mic_audio = np.concatenate(
                mic_chunks, axis=0
            )

            system_audio = np.concatenate(
                system_chunks, axis=0
            )

            sf.write(mic_path, mic_audio, SAMPLE_RATE)
            sf.write(system_path, system_audio, SAMPLE_RATE)

            self.recording_saved.emit(
                mic_path, system_path
            )

        except Exception as exc:
            self.error_occurred.emit(str(exc))


# ==================================================
# VIDEO TRANSCRIPTION WORKER
# ==================================================

class VideoTranscriptionWorker(QThread):

    status_changed = Signal(str)
    transcript_updated = Signal(str)
    transcription_complete = Signal(str, str)
    error_occurred = Signal(str)

    def __init__(self, video_path):
        super().__init__()

        self.video_path = video_path
        self.stop_requested = False

    def run(self):

        try:
            import gc

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            audio_path = os.path.join(
                RECORDINGS_DIR,
                f"uploaded_meeting_{timestamp}.wav"
            )

            transcript_path = os.path.join(
                RECORDINGS_DIR,
                f"uploaded_transcript_{timestamp}.txt"
            )

            # STEP 1: Extract audio from video
            self.status_changed.emit(
                "Extracting audio from uploaded video..."
            )

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

            command = [
                ffmpeg, "-y",
                "-i", self.video_path,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                audio_path
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(
                    "Audio extraction failed:\n"
                    + result.stderr[-2000:]
                )

            if not os.path.exists(audio_path):
                raise RuntimeError(
                    "Extracted audio file was not created."
                )

            # STEP 2: Speaker diarization
            self.status_changed.emit(
                "Loading speaker diarization model..."
            )

            audio, sample_rate = sf.read(
                audio_path,
                dtype="float32"
            )

            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)

            waveform = torch.from_numpy(audio).unsqueeze(0)

            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1"
            )

            pipeline.to(torch.device("cpu"))

            self.status_changed.emit(
                "Identifying meeting speakers..."
            )

            diarization_output = pipeline(
                {
                    "waveform": waveform,
                    "sample_rate": sample_rate
                },
                num_speakers=2
            )

            diarization_turns = [
                (turn, str(speaker))
                for turn, speaker
                in diarization_output.speaker_diarization
            ]

            print("\n===== SPEAKER DIARIZATION DEBUG =====")

            for turn, speaker in diarization_turns:
                print(
                    f"{turn.start:.2f}s - "
                    f"{turn.end:.2f}s : {speaker}"
                )

            print("===== END DEBUG =====\n")

            # Release diarization model memory
            del pipeline
            del diarization_output
            del waveform
            del audio
            gc.collect()

            # Provisional speaker names based on our test
            speaker_names = {
                "SPEAKER_00": "Committee Coordinator",
                "SPEAKER_01": "Student Representative"
            }

            # STEP 3: Load Faster-Whisper on CPU
            self.status_changed.emit(
                "Loading Faster-Whisper Small model..."
            )

            model = WhisperModel(
                "small",
                device="cpu",
                compute_type="int8"
            )

            self.status_changed.emit(
                "Transcribing meeting audio..."
            )

            segments, info = model.transcribe(
                audio_path,
                language=None,
                task="transcribe",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=True,
                initial_prompt=(
                    "Tamil-English code-switching conversation. "
                    "வணக்கம். Today நம்ம previous meetings-ல "
                    "discuss பண்ண complaints-க்கு final status "
                    "update பண்ணப்போறோம். "
                    "Use Tamil script for Tamil words and English "
                    "script for English words. Do not translate."
                )
            )

            transcript_lines = []
            detected_language = info.language

            self.transcript_updated.emit(
                f"Detected language: {detected_language}\n\n"
            )

            # STEP 4: Match transcript segments with speakers
            for segment in segments:

                if self.stop_requested:
                    self.status_changed.emit(
                        "Transcription stopped."
                    )
                    return

                start_seconds = segment.start
                end_seconds = segment.end

                best_speaker = "UNKNOWN"
                best_overlap = 0.0

                for turn, speaker_id in diarization_turns:

                    overlap = max(
                        0.0,
                        min(end_seconds, turn.end)
                        - max(start_seconds, turn.start)
                    )

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = speaker_id

                speaker_name = speaker_names.get(
                    best_speaker,
                    best_speaker
                )

                text = segment.text.strip()

                if not text:
                    continue

                start = self.format_time(start_seconds)
                end = self.format_time(end_seconds)

                line = (
                    f"[{start} - {end}] "
                    f"{speaker_name}: {text}"
                )

                transcript_lines.append(line)

                self.transcript_updated.emit(
                    line + "\n"
                )

            # STEP 5: Save transcript
            if not transcript_lines:
                raise RuntimeError(
                    "No speech detected in the meeting."
                )

            with open(
                transcript_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write("Committee Meeting Transcript\n")
                file.write(
                    f"Detected language: {detected_language}\n\n"
                )

                file.write("\n".join(transcript_lines))

            self.status_changed.emit(
                "Speaker-based transcription completed!"
            )

            self.transcription_complete.emit(
                transcript_path,
                audio_path
            )

        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def format_time(self, seconds):

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:05.2f}"
        )


# ==================================================
# MAIN DESKTOP APPLICATION
# ==================================================

class CommitteeMeetingAdvisor(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Committee Meeting Advisor"
        )

        self.resize(1050, 750)

        self.audio_worker = None
        self.video_worker = None

        self.meeting_started_at = None

        self.setup_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(
            self.update_elapsed_time
        )

    def setup_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(25, 20, 25, 20)

        title = QLabel(
            "Committee Meeting Advisor"
        )

        title.setStyleSheet(
            "font-size: 28px; font-weight: bold; "
            "color: #243B53;"
        )

        subtitle = QLabel(
            "Live meeting recording and video-based meeting analysis"
        )

        subtitle.setStyleSheet(
            "font-size: 14px; color: #627D98;"
        )

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        self.tabs = QTabWidget()

        self.live_tab = QWidget()
        self.upload_tab = QWidget()

        self.tabs.addTab(
            self.live_tab,
            "Live Meeting"
        )

        self.tabs.addTab(
            self.upload_tab,
            "Upload Meeting Video"
        )

        self.setup_live_tab()
        self.setup_upload_tab()

        main_layout.addWidget(self.tabs)

    # ----------------------------------------------
    # LIVE MEETING TAB
    # ----------------------------------------------

    def setup_live_tab(self):

        layout = QVBoxLayout(self.live_tab)

        status_frame = QFrame()
        status_frame.setStyleSheet(
            "QFrame { background: #F0F4F8; "
            "border-radius: 12px; }"
        )

        status_layout = QVBoxLayout(status_frame)

        self.live_status = QLabel(
            "Status: Ready"
        )

        self.live_time = QLabel(
            "Meeting duration: 00:00:00"
        )

        self.mic_label = QLabel(
            "Microphone: Ready"
        )

        self.system_label = QLabel(
            "System audio: Ready"
        )

        status_layout.addWidget(self.live_status)
        status_layout.addWidget(self.live_time)
        status_layout.addWidget(self.mic_label)
        status_layout.addWidget(self.system_label)

        layout.addWidget(status_frame)

        buttons = QHBoxLayout()

        self.start_button = QPushButton(
            "Start Meeting"
        )

        self.stop_button = QPushButton(
            "Stop Meeting"
        )

        self.start_button.clicked.connect(
            self.start_meeting
        )

        self.stop_button.clicked.connect(
            self.stop_meeting
        )

        self.stop_button.setEnabled(False)

        self.start_button.setStyleSheet(
            "background:#16803C;color:white;"
            "padding:12px;font-weight:bold;"
        )

        self.stop_button.setStyleSheet(
            "background:#C0392B;color:white;"
            "padding:12px;font-weight:bold;"
        )

        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)

        layout.addLayout(buttons)

        layout.addWidget(
            QLabel("Live Meeting Transcript")
        )

        self.live_transcript = QTextEdit()
        self.live_transcript.setReadOnly(True)

        self.live_transcript.setPlaceholderText(
            "Live transcript will appear here "
            "when speech recognition is connected."
        )

        layout.addWidget(
            self.live_transcript
        )

    # ----------------------------------------------
    # UPLOAD VIDEO TAB
    # ----------------------------------------------

    def setup_upload_tab(self):

        layout = QVBoxLayout(self.upload_tab)

        heading = QLabel(
            "Upload a Recorded Committee Meeting"
        )

        heading.setStyleSheet(
            "font-size: 20px; font-weight: bold;"
            "color: #243B53;"
        )

        description = QLabel(
            "Select a recorded meeting video to extract "
            "audio and generate a timestamped transcript."
        )

        description.setWordWrap(True)

        layout.addWidget(heading)
        layout.addWidget(description)

        self.selected_video_label = QLabel(
            "No video selected."
        )

        self.selected_video_label.setWordWrap(True)

        layout.addWidget(
            self.selected_video_label
        )

        self.browse_button = QPushButton(
            "Browse Video"
        )

        self.browse_button.clicked.connect(
            self.select_video
        )

        layout.addWidget(
            self.browse_button
        )

        self.transcribe_button = QPushButton(
            "Generate Transcript"
        )

        self.transcribe_button.setEnabled(False)

        self.transcribe_button.clicked.connect(
            self.start_video_transcription
        )

        self.transcribe_button.setStyleSheet(
            "background:#2457A7;color:white;"
            "padding:12px;font-weight:bold;"
        )

        layout.addWidget(
            self.transcribe_button
        )

        self.upload_status = QLabel(
            "Status: Waiting for video"
        )

        self.upload_status.setStyleSheet(
            "font-weight:bold;color:#486581;"
        )

        layout.addWidget(
            self.upload_status
        )

        layout.addWidget(
            QLabel("Generated Meeting Transcript")
        )

        self.upload_transcript = QTextEdit()
        self.upload_transcript.setReadOnly(True)

        self.upload_transcript.setPlaceholderText(
            "Your timestamped transcript will appear here."
        )

        layout.addWidget(
            self.upload_transcript
        )

        self.selected_video_path = None

    # ----------------------------------------------
    # LIVE RECORDING FUNCTIONS
    # ----------------------------------------------

    def start_meeting(self):

        if self.audio_worker and self.audio_worker.isRunning():
            return

        self.audio_worker = AudioRecordingWorker(
            mic_device_id=1
        )

        self.audio_worker.status_changed.connect(
            self.update_live_status
        )

        self.audio_worker.recording_saved.connect(
            self.live_recording_complete
        )

        self.audio_worker.error_occurred.connect(
            self.live_recording_error
        )

        self.meeting_started_at = time.monotonic()

        self.live_transcript.clear()

        self.live_transcript.append(
            "Meeting recording started.\n"
            "Live transcription will be connected next."
        )

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.timer.start(1000)

        self.audio_worker.start()

    def stop_meeting(self):

        if self.audio_worker and self.audio_worker.isRunning():

            self.stop_button.setEnabled(False)

            self.live_status.setText(
                "Status: Stopping recording..."
            )

            self.audio_worker.stop_recording()

    def update_live_status(self, message):

        self.live_status.setText(
            "Status: " + message
        )

        if "Recording microphone" in message:

            self.mic_label.setText(
                "Microphone: Recording"
            )

            self.system_label.setText(
                "System audio: Recording"
            )

    def update_elapsed_time(self):

        if self.meeting_started_at is None:
            return

        elapsed = int(
            time.monotonic() - self.meeting_started_at
        )

        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        self.live_time.setText(
            f"Meeting duration: "
            f"{hours:02}:{minutes:02}:{seconds:02}"
        )

    def live_recording_complete(
        self, mic_path, system_path
    ):

        self.timer.stop()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.live_status.setText(
            "Status: Recording saved"
        )

        self.mic_label.setText(
            "Microphone: Saved"
        )

        self.system_label.setText(
            "System audio: Saved"
        )

        self.live_transcript.append(
            "\nBoth recordings saved successfully.\n"
            f"Microphone: {mic_path}\n"
            f"System audio: {system_path}"
        )

        QMessageBox.information(
            self,
            "Recording Saved",
            "Both audio recordings were saved successfully."
        )

    def live_recording_error(self, message):

        self.timer.stop()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.live_status.setText(
            "Status: Recording error"
        )

        QMessageBox.critical(
            self,
            "Recording Error",
            message
        )

    # ----------------------------------------------
    # VIDEO UPLOAD FUNCTIONS
    # ----------------------------------------------

    def select_video(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Meeting Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;"
            "All Files (*.*)"
        )

        if file_path:

            self.selected_video_path = file_path

            self.selected_video_label.setText(
                "Selected video: " + file_path
            )

            self.transcribe_button.setEnabled(True)

            self.upload_status.setText(
                "Status: Video selected"
            )

            self.upload_transcript.clear()

    def start_video_transcription(self):

        if not self.selected_video_path:
            return

        if self.video_worker and self.video_worker.isRunning():
            return

        self.browse_button.setEnabled(False)
        self.transcribe_button.setEnabled(False)

        self.upload_transcript.clear()

        self.video_worker = VideoTranscriptionWorker(
            self.selected_video_path
        )

        self.video_worker.status_changed.connect(
            self.update_upload_status
        )

        self.video_worker.transcript_updated.connect(
            self.show_uploaded_transcript
        )

        self.video_worker.transcription_complete.connect(
            self.video_transcription_complete
        )

        self.video_worker.error_occurred.connect(
            self.video_transcription_error
        )

        self.video_worker.start()

    def update_upload_status(self, message):

        self.upload_status.setText(
            "Status: " + message
        )

    def show_uploaded_transcript(self, text):

        self.upload_transcript.moveCursor(
            QTextCursor.MoveOperation.End
        )

        self.upload_transcript.insertPlainText(text)

    def video_transcription_complete(
        self, transcript_path, audio_path
    ):

        self.browse_button.setEnabled(True)
        self.transcribe_button.setEnabled(True)

        self.upload_status.setText(
            "Status: Transcription completed!"
        )

        QMessageBox.information(
            self,
            "Transcription Complete",
            "Meeting transcript generated successfully.\n\n"
            f"Transcript:\n{transcript_path}\n\n"
            f"Extracted audio:\n{audio_path}"
        )

    def video_transcription_error(self, message):

        self.browse_button.setEnabled(True)
        self.transcribe_button.setEnabled(True)

        self.upload_status.setText(
            "Status: Transcription failed"
        )

        QMessageBox.critical(
            self,
            "Video Processing Error",
            message
        )

    # ----------------------------------------------
    # APPLICATION CLOSE
    # ----------------------------------------------

    def closeEvent(self, event):

        if self.audio_worker and self.audio_worker.isRunning():

            self.audio_worker.stop_recording()
            self.audio_worker.wait(5000)

        if self.video_worker and self.video_worker.isRunning():

            QMessageBox.warning(
                self,
                "Processing in progress",
                "Please wait for video transcription to finish."
            )

            event.ignore()
            return

        event.accept()


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = CommitteeMeetingAdvisor()
    window.show()

    sys.exit(app.exec())
