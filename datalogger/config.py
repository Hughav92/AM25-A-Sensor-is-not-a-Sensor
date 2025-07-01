"""
config.py
---------
Configuration file for the datalogger system.

Contains global settings, stream options, device addresses, and experiment parameters.
"""

import pyaudio

### Main ###

settings = {
    "SIFI": False,
    "MYO": False,
    "QUALISYS": False,
    "AUDIO": False
}

stream_settings = {
    "SIFI":False,
    "MYO":False
}

### Myo ###

myo_ip = "127.0.0.1"
myo_port = 16000

### SIFIBAND ###

IMU = True
EMG = True
RETURN_EULER = True
EULER_ANGLES = "rad"
RETURN_ANGULAR_VELOCITY = True
STREAM_G = True

sifi_ip = "127.0.0.1"
sifi_port = 16000

sifi_imu_sr = 50

### Qualisys ### 

rigid_body_addresses = ["Ambient_Clapper", "Myo", "SiFiBand"]
command_ip = "172.22.40.9"
command_port = 22225
server_ip = "172.22.40.8"
server_port = 45455

### audio ###

input_audio_device = 2
sample_format = pyaudio.paFloat32
channels = 2  # Set to stereo
fs = 48000
chunk = int(fs * 0.1)

### waveform generator ###

frequency = 10
waveform = "sine"
ratio = 0.5
duration = 1
initialisation_wait = 1
voltage_low = 0
voltage_high = 0.16 #2.9 

generator_pulses = True
no_pulses = 500
pulse_gap = 10