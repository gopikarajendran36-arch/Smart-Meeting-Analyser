
import sounddevice as sd
import soundcard as sc

print("\n========== MICROPHONE DEVICES ==========")

devices = sd.query_devices()

for index, device in enumerate(devices):
    if device["max_input_channels"] > 0:
        print(
            f"ID: {index} | "
            f"Name: {device['name']} | "
            f"Input channels: {device['max_input_channels']}"
        )

print("\n========== SYSTEM AUDIO DEVICES ==========")

for speaker in sc.all_speakers():
    print("Speaker:", speaker.name)

print("\n========== LOOPBACK MICROPHONES ==========")

for mic in sc.all_microphones(include_loopback=True):
    print(
        "Name:", mic.name,
        "| Loopback:", mic.isloopback
    )
