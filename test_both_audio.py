
import sounddevice as sd
import soundcard as sc
import threading

mic_data = []
system_data = []
errors = []

def record_microphone():
    try:
        print("Microphone recording started...")

        data = sd.rec(
            int(5 * 16000),
            samplerate=16000,
            channels=1,
            device=1,
            dtype="float32"
        )

        sd.wait()
        mic_data.append(data)

        print("Microphone recording completed.")

    except Exception as e:
        errors.append("Microphone: " + str(e))
        print("Microphone error:", e)


def record_system_audio():
    try:
        print("System audio recording started...")

        speaker = sc.default_speaker()

        # Get the loopback microphone for the default speaker
        loopback_mic = sc.get_microphone(
            id=str(speaker.name),
            include_loopback=True
        )

        with loopback_mic.recorder(samplerate=16000) as recorder:
            data = recorder.record(numframes=16000 * 5)

        system_data.append(data)

        print("System audio recording completed.")

    except Exception as e:
        errors.append("System audio: " + str(e))
        print("System audio error:", e)


mic_thread = threading.Thread(target=record_microphone)
system_thread = threading.Thread(target=record_system_audio)

print("\nStarting both recordings for 5 seconds...")

mic_thread.start()
system_thread.start()

mic_thread.join()
system_thread.join()

print("\n========== RESULTS ==========")

if mic_data:
    print("Microphone shape:", mic_data[0].shape)
else:
    print("Microphone recording failed.")

if system_data:
    print("System audio shape:", system_data[0].shape)
else:
    print("System audio recording failed.")

if not errors:
    print("\nBoth audio sources recorded successfully!")
else:
    print("\nErrors encountered:")
    for error in errors:
        print(error)
