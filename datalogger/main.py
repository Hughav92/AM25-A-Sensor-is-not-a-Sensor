"""
main.py
-------
Main control script for the datalogger system.

This script provides an interactive command-line interface for starting, stopping, and managing synchronized data logging and streaming from multiple sources:
- SiFiBand (EMG/IMU)
- Myo Armband (EMG/IMU)
- Qualisys motion capture
- Audio recording
- Waveform generator control

Features:
- Keyboard-driven control for starting/stopping recordings, toggling sources, logging events, and setting parameters.
- Each data source runs in its own process for parallel acquisition.
- Supports both logging to disk and real-time streaming (OSC) for SiFi and Myo.
- Event logging and waveform generator event tracking.
- Audio device selection and waveform parameter configuration.

Usage:
    Run this script and use the keyboard commands printed at startup to control the system.
    Data and event logs are saved in the 'logs' directory with a timestamped session name.

    Connects to a Myo device using the dongless-myo library. This can be found here: https://github.com/iomz/dl-myo
    The following scripts need to be placed in the myo directory:
    - __init__.py
    - commands.py
    - constants.py
    - core.py
    - profile.py
    - types.py
    - version.py
    Connects to a BitScope Micro device using the scopething library. This can be found here: https://github.com/jonathanhogg/scopething
    The following scripts need to be placed in the scopething directory:
    - __init__.py
    - analysis.py
    - scope.py
    - streams.py
    - test.py
    - utils.py
    Requires sifibridge.exe to be placed in this directory to connect with the SiFiBand.
"""

from multiprocessing import Process, Event, Queue
from sifiband_log import *
from myo_log import myo_main, myo_cleanup
from audio_record import log_audio, audio_cleanup
from qualisys_log import qualisys_main, qualisys_cleanup
from config import settings, stream_settings, myo_ip, myo_port, sifi_ip, sifi_port, input_audio_device, frequency, duration, ratio, waveform, IMU, EMG, voltage_low, voltage_high, generator_pulses, no_pulses, pulse_gap, initialisation_wait
from generate_waveform import generate_waveform
from datetime import datetime
import keyboard
import time
import pyaudio
import asyncio
import msvcrt
import sys

def flush_input():
    """
    Flush any pending keyboard input (Windows only).
    
    Args:
        None
    Returns:
        None
    """
    while msvcrt.kbhit():
        msvcrt.getch()

def start_recording(log_name, stop_event):
    """
    Starts the recording processes based on the settings.
    
    Args:
        log_name (str): The name of the log file/session.
        stop_event (Event): Multiprocessing event to signal stop.
    Returns:
        list: A list of started processes.
    """

    global input_audio_device

    processes = []
    # Start each enabled process and append to the list
    if settings["SIFI"]:
        p_sb = Process(target=log_sifiband, kwargs={"log_name": log_name, "sifi_stream":stream_settings["SIFI"], "stop_event":stop_event, "sifi_ip":sifi_ip, "sifi_port":sifi_port, "IMU":IMU, "EMG":EMG}, daemon=True)
        p_sb.start()
        processes.append(p_sb)
    if settings["MYO"]:
        p_myo = Process(target=myo_main, kwargs={"log_name": log_name, "myo_stream":stream_settings["MYO"], "stop_event":stop_event, "myo_ip":myo_ip, "myo_port":myo_port}, daemon=True)
        p_myo.start()
        processes.append(p_myo)
    if settings["QUALISYS"]:
        p_qual = Process(target=qualisys_main, kwargs={"log_name": log_name, "stop_event":stop_event}, daemon=True)
        p_qual.start()
        processes.append(p_qual)
    if settings["AUDIO"]:
        p_audio = Process(target=log_audio, kwargs={"log_name": log_name, "input_audio_device":input_audio_device, "stop_event":stop_event}, daemon=True)
        p_audio.start()
        processes.append(p_audio)

    return processes

def write_event_log(event_log, log_name):
    """
    Save the event log as both a joblib and CSV file.
    
    Args:
        event_log (dict): Dictionary of event names and timestamps.
        log_name (str): The name of the log file/session.
    Returns:
        None
    """

    if not os.path.exists(os.path.join('logs', log_name, 'event_log')):
        os.makedirs(os.path.join('logs', log_name, 'event_log'))
        print(f"Folder created at {os.path.join('logs', log_name, 'event_log')}")
    else:
        print(f"Folder already exists at {os.path.join('logs', log_name, 'event_log')}")

    joblib.dump(event_log, os.path.join('logs', log_name, 'event_log', 'event_log.joblib'))
    log_df = pd.DataFrame(event_log.items(), columns=["Event", "Event Time"])
    log_df.to_csv(os.path.join('logs', log_name, 'event_log', 'event_log.csv'), index=False)

def stop_recording(processes, log_name, stop_event, event_log):
    """
    Stops the recording processes and performs cleanup.
    
    Args:
        processes (list): The list of processes to stop.
        log_name (str): The name of the log file/session.
        stop_event (Event): Multiprocessing event to signal stop.
        event_log (dict): Dictionary of event names and timestamps.
    Returns:
        None
    """

    stop_event.set()
    print("Stopping Recording")
    time.sleep(1)

    for p in processes:
        p.terminate()
    for p in processes:
        p.join()

    write_event_log(event_log, log_name)

    if settings["SIFI"]:
        sifiband_cleanup(log_name, IMU, EMG)
    if settings["MYO"]:
        myo_cleanup(log_name)
    if settings["QUALISYS"]:   
        qualisys_cleanup(log_name)
    if settings["AUDIO"]:
        audio_cleanup(log_name)

def print_audio_devices():
    """
    Prints the available audio input devices.
    
    Args:
        None
    Returns:
        None
    """
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"Device {i}: {device_info['name']} - Channels: {device_info['maxInputChannels']}")
    p.terminate()

def set_audio_device():
    """
    Sets the input audio device based on user input.
    
    Args:
        None
    Returns:
        None
    """
    global input_audio_device
    print(f"Current audio input device: {input_audio_device}")
    flush_input()
    time.sleep(0.1)
    device_index = input("Enter the device index: ")
    input_audio_device = int(device_index)
    print(f"Audio input device set to {input_audio_device}")

def set_waveform_generator():
    """
    Interactive menu for setting waveform generator parameters.
    
    Args:
        None
    Returns:
        None
    """

    global waveform, frequency, duration, ratio, voltage_low, voltage_high, generator_pulses, no_pulses, pulse_gap

    print("Current Waveform Parameters:")
    print(f"Waveform -> {waveform}, Frequency -> {frequency}, Duty Cycle Ratio -> {ratio}, Duration -> {duration}, Voltage High -> {voltage_high}, Voltage Low -> {voltage_low}, Generator Pulses -> {generator_pulses}")
    if generator_pulses:
        print(f"Number of Generator Pulses -> {no_pulses}, Duration between pulses -> {pulse_gap}")
    print("Press to set: 'w' for waveform, 'f' for frequency, 'r' for duty cycle ratio, 'd' for duration, 'h' for voltage high, 'l' for voltage low")
    print("Press 'g' to set generator pulsing parameters")
    print("Press 'p' to see status")
    print("Press 'e' to exit waveform settings")
    time.sleep(0.1)

    waveform_settings = True

    while waveform_settings:

        if keyboard.is_pressed('w'):
            print(f"Current waveform {waveform}")
            time.sleep(0.1)
            flush_input()
            waveform = input("Enter waveform ('sine', 'triangle', 'exponential', 'square'): ")
            waveform = str(waveform)
            print(f"Waveform is set to {waveform}")
            time.sleep(0.1)
        if keyboard.is_pressed('f'):
            print(f"Current frequency: {frequency}")
            time.sleep(0.1)
            flush_input()
            frequency = input("Enter frequency: ")
            frequency = float(frequency)
            print(f"Frequency is set to {frequency}")
            time.sleep(0.1)
        if keyboard.is_pressed('r'):
            print(f"Current duty cycle ratio {ratio}")
            time.sleep(0.1)
            flush_input()
            ratio = input("Enter ratio: ")
            ratio = float(ratio)
            print(f"Duty cycle ratio is set to {ratio}")
            time.sleep(0.1)
        if keyboard.is_pressed('d'):
            print(f"Current duration: {duration}")
            time.sleep(0.1)
            flush_input()
            duration = input("Enter duration: ")
            duration = float(duration)
            print(f"Duration is set to {duration}")
            time.sleep(0.1)
        if keyboard.is_pressed('h'):   
            print(f"Current voltage high {voltage_high}")
            time.sleep(0.1)
            flush_input()
            voltage_high = input("Enter voltage high: ")
            voltage_high = float(voltage_high)
            print(f"Voltage high is set to {voltage_high}")
            time.sleep(0.1)
        if keyboard.is_pressed('l'):
            print(f"Current voltage low {voltage_low}")
            time.sleep(0.1)
            flush_input()
            voltage_low = input("Enter voltage low: ")
            voltage_low = float(voltage_low)
            print(f"Voltage low is set to {voltage_low}")
            time.sleep(0.1)
        if keyboard.is_pressed('g'):
            print(f"Current Generator Pulses: {generator_pulses}")
            if generator_pulses:
                print(f"Current number of pulses is {no_pulses}")
                print(f"Current duration between pulses is {pulse_gap} seconds")
            time.sleep(0.1)
            flush_input()
            generator_pulses = input("Enter generator pulses (bool): ")
            generator_pulses = bool(generator_pulses)
            if generator_pulses:
                no_pulses = input("Enter number of generator pulses: ")
                no_pulses = float(no_pulses)
                pulse_gap = input("Enter duration between pulses (seconds): ")
                pulse_gap = float(pulse_gap)
        if keyboard.is_pressed('p'):
            print(f"Waveform -> {waveform}, Frequency -> {frequency}, Duty Cycle Ratio -> {ratio}, Duration -> {duration}, Voltage High -> {voltage_high}, Voltage Low -> {voltage_low}")
            time.sleep(0.1)
        if keyboard.is_pressed('e'):
            print("Exiting waveform settings...")
            waveform_settings = False
            time.sleep(0.1)

def log_event(event_log, event_time=None, event_name=None):
    """
    Log an event with a timestamp and optional name.
    
    Args:
        event_log (dict): Dictionary to store events.
        event_time (float, optional): Timestamp for the event. Defaults to current time.
        event_name (str, optional): Name for the event. Defaults to empty string.
    Returns:
        None
    """
    if event_time is None:
        event_time = time.time()
    if event_name is None:
        event_name = ""
    event_log[event_name] = event_time
    if event_name == "":
        print(f"Logged Event: {event_time}")
    else:
        print(f"Logged Event: {event_name} at {event_time}")

def print_help():
    """
    Prints the available keyboard commands for the user interface.
    
    Args:
        None
    Returns:
        None
    """

    print("Press 'r' to start recording and 't' to stop recording.")
    print("Press 's' for SIFI, 'm' for MYO, 'q' for QUALISYS, 'a' for AUDIO, and 'p' for status.")
    print("Press 'x' for SIFI stream and 'n' for MYO Stream")
    print("Press 'v' to view audio devices and set input device")
    print("Press 'j' to set log name")
    print("Press 'l' to log an event while recording in process")
    print("Press 'w' to set waveform generator parameters")
    print("Press 'g' to generate a waveform, 'b' to stop waveform generator")
    print("Press 'h' to print these commands")
    print("Press 'e' to exit the program.")

def main():
    """
    Main function to handle user input and control the recording processes.
    
    Args:
        None
    Returns:
        None
    """
    processes = []

    stop_event = Event()
    name_changed = False

    event_log = {}

    p_gen = None
    pulse_count = 0
    event_count = 0

    print_help()

    while True:
        if keyboard.is_pressed('r') and not processes:
            pulse_count = 0
            if not name_changed:
                log_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            print("Recording started...")
            processes = start_recording(log_name, stop_event)
            time.sleep(0.1)
        elif keyboard.is_pressed('t') and processes:
            stop_recording(processes, log_name, stop_event, event_log)
            print("Recording stopped.")
            processes = []
            stop_event = Event()
            name_changed = False
            event_log = {}
            time.sleep(0.1)

        if keyboard.is_pressed('l') and processes:
            log_event(event_log, event_time=time.time(), event_name=f"Event {str(event_count).zfill(5)}")
            event_count += 1
            time.sleep(0.1)

        if keyboard.is_pressed('g') and p_gen is None:
            if generator_pulses:
                counter = 0
                while counter < no_pulses:
                    output_queue = Queue()
                    task_queue = Queue()
                    p_gen = Process(target=generate_waveform,
                                    kwargs={"f": frequency,
                                            "duration":duration,
                                            "initialisation_wait":initialisation_wait,
                                            "ratio":ratio,
                                            "waveform":waveform,
                                            "low":voltage_low,
                                            "high":voltage_high,
                                            "output_queue":output_queue,
                                            "task_queue":task_queue},
                                    daemon=True)
                    p_gen.start()
                    print(f"Generating Waveform with parameters frequency -> {frequency}, duration -> {duration}, initialisation_wait -> {initialisation_wait}, ratio -> {ratio}, waveform -> {waveform}, low -> {voltage_low}, high -> {voltage_high}")
                    s = time.perf_counter()
                    if p_gen is not None:
                        while output_queue.empty():
                            pass
                        if not output_queue.empty():
                            gen_times = output_queue.get()
                            gen_start, gen_end = gen_times
                            log_event(event_log, event_time=gen_start, event_name=f"Waveform Pulse Start {str(pulse_count).zfill(5)}")
                            log_event(event_log, event_time=gen_end, event_name=f"Waveform Pulse End {str(pulse_count).zfill(5)}")
                            output_queue = Queue()
                            pulse_count += 1
                            print(f"Pulse length: {gen_end-gen_start}")
                            p_gen = None
                    while time.perf_counter()-s < pulse_gap:
                        pass
                    counter += 1
            else:
                output_queue = Queue()
                task_queue = Queue()
                p_gen = Process(target=generate_waveform,
                                kwargs={"f": frequency,
                                        "duration":duration,
                                        "initialisation_wait":initialisation_wait,
                                        "ratio":ratio,
                                        "waveform":waveform,
                                        "low":voltage_low,
                                        "high":voltage_high,
                                        "output_queue":output_queue,
                                        "task_queue":task_queue},
                                daemon=True)
                p_gen.start()
                print(f"Generating Waveform with parameters frequency -> {frequency}, duration -> {duration}, initialisation_wait -> {initialisation_wait}, ratio -> {ratio}, waveform -> {waveform}, low -> {voltage_low}, high -> {voltage_high}")
            time.sleep(0.1)

        if keyboard.is_pressed('b') and p_gen is not None:
            task_queue.put("STOP")
            time.sleep(0.1)

        if p_gen is not None:
            if not output_queue.empty():
                gen_times = output_queue.get()
                gen_start, gen_end = gen_times
                log_event(event_log, event_time=gen_start, event_name=f"Waveform Pulse Start {str(pulse_count).zfill(5)}")
                log_event(event_log, event_time=gen_end, event_name=f"Waveform Pulse End {str(pulse_count).zfill(5)}")
                output_queue = Queue()
                pulse_count += 1
                print(f"Pulse length: {gen_end-gen_start}")
                p_gen = None
                time.sleep(0.1)

        if keyboard.is_pressed('h'):
            print_help()
            time.sleep(0.1)

        if not processes:
            if keyboard.is_pressed('s'):
                settings["SIFI"] = not settings["SIFI"]
                print(f"SIFI toggled to {settings['SIFI']}")
                time.sleep(0.1)
            if keyboard.is_pressed('x'):
                stream_settings["SIFI"] = not stream_settings["SIFI"]
                print(f"SIFI stream toggled to {stream_settings['SIFI']}")
                time.sleep(0.1)
            if keyboard.is_pressed('m'):
                settings["MYO"] = not settings["MYO"]
                print(f"MYO toggled to {settings['MYO']}")
                time.sleep(0.1)
            if keyboard.is_pressed('n'):
                stream_settings["MYO"] = not stream_settings["MYO"]
                print(f"MYO stream toggled to {stream_settings['MYO']}")
                time.sleep(0.1)
            if keyboard.is_pressed('q'):
                settings["QUALISYS"] = not settings["QUALISYS"]
                print(f"QUALISYS toggled to {settings['QUALISYS']}")
                time.sleep(0.1)
            if keyboard.is_pressed('a'):
                settings["AUDIO"] = not settings["AUDIO"]
                print(f"AUDIO toggled to {settings['AUDIO']}")
                time.sleep(0.1)
            if keyboard.is_pressed('p'):
                print(f"Status -> SIFI: {settings['SIFI']}, MYO: {settings['MYO']}, QUALISYS: {settings['QUALISYS']}, AUDIO: {settings['AUDIO']}")
                print(f"Stream status -> SIFI: {stream_settings['SIFI']}, MYO: {stream_settings['MYO']}")
                time.sleep(0.1)
            if keyboard.is_pressed('v'):
                print_audio_devices()
                set_audio_device()
                time.sleep(0.1)
            if keyboard.is_pressed('w'):
                set_waveform_generator()
                time.sleep(0.1)
            if keyboard.is_pressed('j'):
                try:
                    print(f"Current log name: {log_name}")
                except NameError:
                    print(f"Current log name: datetime")
                time.sleep(0.1)
                flush_input()
                log_name = input("Enter log name or j for datetime: ")
                date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                log_name = f"{date}_{log_name}"
                print(f"New log name: {log_name}")
                name_changed = True
                time.sleep(0.1)
            if keyboard.is_pressed('e'):
                print("Exiting program.")
                break

        time.sleep(0.1)

if __name__ == "__main__":
    main()
