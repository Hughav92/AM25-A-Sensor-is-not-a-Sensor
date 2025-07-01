"""
audio_record.py
---------------
Handles audio recording for the datalogger system.

Provides functions to start, stop, and clean up audio recording processes, saving audio data to disk.
"""

import pyaudio
import numpy as np
from datetime import datetime
import time
import os
import wave
import pandas as pd
import joblib
import itertools
from config import sample_format, channels, fs, chunk, input_audio_device
from multiprocessing import Event

# Global variables for storing audio data
audio_list = []
timestamp_list = []
chunk_vals = []
dump_count = 0

def write_audio(audio_list, filename, fs=48000, channels=2):
    """
    Writes the audio data to a WAV file.
    
    Args:
        audio_list (list): The list of audio data chunks.
        filename (str): The name of the output WAV file.
        fs (int): The sampling rate.
        channels (int): The number of audio channels.
    """
    audio_data = np.concatenate(audio_list, axis=0)
    audio_data = audio_data.reshape(-1, channels)  # Ensure correct stereo shape
    audio_data_int16 = np.int16(audio_data * 32767)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_data_int16.tobytes())

def log_audio(log_name=None, print_devices=False, input_audio_device=input_audio_device, stop_event=None):
    """
    Logs audio data from the input device.
    
    Args:
        log_name (str): The name of the log file.
        print_devices (bool): Whether to print the available audio devices.
    """
    global audio_list, timestamp_list, chunk_vals, dump_count

    if print_devices:
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            print(f"Device {i}: {device_info['name']} - Channels: {device_info['maxInputChannels']}")
        p.terminate()

    if log_name is None:
        log_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if not os.path.exists(os.path.join("logs", log_name, "audio")):
        os.makedirs(os.path.join("logs", log_name, "audio"))
        print(f"Folder created at {os.path.join('logs', log_name, 'audio')}")
    else:
        print(f"Folder already exists at {os.path.join('logs', log_name, 'audio')}")

    p = pyaudio.PyAudio()
    chunk_count = 0
    print(f"Using audio input device: {input_audio_device}")  # Print to confirm it's the right one
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input_device_index=input_audio_device,
                    input=True)

    try:
        stop_triggered = False
        while True:
            data = stream.read(chunk, exception_on_overflow=False)
            data = np.frombuffer(data, dtype=np.float32)
            audio_list.append(data)
            timestamp_list.append(time.time())
            chunk_vals.append(chunk_count * chunk)

            if chunk_count % 10000 == 0:
                joblib.dump(audio_list, os.path.join("logs", log_name, "audio", f"audio_data_{str(dump_count).zfill(5)}.joblib"))
                joblib.dump(timestamp_list, os.path.join("logs", log_name, "audio", f"audio_timestamps_{str(dump_count).zfill(5)}.joblib"))
                joblib.dump(chunk_vals, os.path.join("logs", log_name, "audio", f"audio_sample_vals_{str(dump_count).zfill(5)}.joblib"))

                audio_list = []
                timestamp_list = []
                chunk_vals = []
                dump_count += 1

            chunk_count += 1

            if stop_event.is_set() and not stop_triggered:
                stop_triggered = True
                if audio_list:
                    joblib.dump(audio_list, os.path.join("logs", log_name, "audio", f"audio_data_{str(dump_count).zfill(5)}.joblib"))
    
    except KeyboardInterrupt:
        joblib.dump(audio_list, os.path.join("logs", log_name, "audio", f"audio_data_{str(dump_count).zfill(5)}.joblib"))

        audio_output = []
        timestamp_output = []
        chunk_vals_output = []

        for file in sorted(os.listdir(os.path.join("logs", log_name, "audio"))):
            if "audio_data" in file:
                audio_data = joblib.load(os.path.join("logs", log_name, "audio", file))
                audio_output.append(audio_data)
            if "timestamps" in file:
                timestamp_data = joblib.load(os.path.join("logs", log_name, "audio", file))
                timestamp_output.append(timestamp_data)
            if "sample_vals" in file:
                chunk_vals_data = joblib.load(os.path.join("logs", log_name, "audio", file))
                chunk_vals_output.append(chunk_vals_data)

        audio_output = list(itertools.chain.from_iterable(audio_output))
        timestamp_output = list(itertools.chain.from_iterable(timestamp_output))
        chunk_vals_output = list(itertools.chain.from_iterable(chunk_vals_output))
        
        write_audio(audio_output, os.path.join("logs", log_name, "audio", "audio.wav"), fs, channels)

        audio_dict = {"sample": chunk_vals_output, "timestamp": timestamp_output}
        audio_df = pd.DataFrame(audio_dict)
        audio_df.to_csv(os.path.join("logs", log_name, "audio", "audio_timestamps.csv"), index=False)

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def audio_cleanup(log_name):
    """
    Cleans up and saves the audio data.
    
    Args:
        log_name (str): The name of the log file.
    """

    audio_output = []
    timestamp_output = []
    chunk_vals_output = []

    for file in sorted(os.listdir(os.path.join("logs", log_name, "audio"))):
        if "audio_data" in file:
            audio_data = joblib.load(os.path.join("logs", log_name, "audio", file))
            audio_output.append(audio_data)
        if "timestamps" in file:
            timestamp_data = joblib.load(os.path.join("logs", log_name, "audio", file))
            timestamp_output.append(timestamp_data)
        if "sample_vals" in file:
            chunk_vals_data = joblib.load(os.path.join("logs", log_name, "audio", file))
            chunk_vals_output.append(chunk_vals_data)

    audio_output = list(itertools.chain.from_iterable(audio_output))
    timestamp_output = list(itertools.chain.from_iterable(timestamp_output))
    chunk_vals_output = list(itertools.chain.from_iterable(chunk_vals_output))
    
    write_audio(audio_output, os.path.join("logs", log_name, "audio", "audio.wav"), fs, channels)

    audio_dict = {"sample": chunk_vals_output, "timestamp": timestamp_output}
    audio_df = pd.DataFrame(audio_dict)
    audio_df.to_csv(os.path.join("logs", log_name, "audio", "audio_timestamps.csv"), index=False)
