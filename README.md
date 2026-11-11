\# Smart Meeting Analyser



An AI-powered desktop application that helps analyze committee meetings through audio/video transcription, speaker diarization, and meeting transcript generation.



\## Features



\- Upload recorded meeting videos.

\- Extract audio from video files.

\- Convert speech into text using Faster-Whisper.

\- Support Tamil-English mixed-language transcription.

\- Generate timestamped transcripts.

\- Identify different speakers using Pyannote Audio.

\- Display speaker-labelled transcripts.

\- Record microphone and system audio during live meetings.

\- Export generated transcripts.



\## Technology Stack



\- Python 3.12

\- PySide6

\- Faster-Whisper

\- Pyannote Audio

\- PyTorch

\- FFmpeg

\- NumPy

\- SoundDevice

\- SoundCard

\- SoundFile



\## Project Structure



```text

Committee-Meeting-Advisor/

├── desktop\_app.py

├── test\_transcription.py

├── test\_diarization.py

├── test\_audio\_devices.py

├── test\_both\_audio.py

├── requirements.txt

├── .gitignore

└── README.md

```



\## Installation



1\. Install Python 3.12.

2\. Clone this repository.

3\. Create a virtual environment:



```bash

python -m venv diarization\_env

```



4\. Activate the environment on Windows:



```powershell

.\\diarization\_env\\Scripts\\Activate.ps1

```



5\. Install the dependencies:



```bash

python -m pip install -r requirements.txt

```



6\. Configure FFmpeg and Hugging Face access for the Pyannote speaker diarization model.



\## Run the Application



```bash

python desktop\_app.py

```



\## Speaker Diarization Setup



The speaker diarization feature uses Pyannote Audio.



A Hugging Face account and access authorization for the required diarization model may be necessary. Configure your access token locally. Never commit tokens or credentials to GitHub.



\## Team Project



Developed as a final-year B.Tech project.



\## Note



This project is under development. Transcription accuracy, especially for Tamil-English code-switching, and speaker diarization are still being improved.

