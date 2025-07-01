"""
sificonnect.py
--------------
Streams SiFiBand EMG and IMU data via OSC (Open Sound Control) to a specified IP and port.

- Connects to a SiFiBand using sifi_bridge_py. Requires the sifibridge.exe in the same directory.
- Streams EMG and IMU data in real time.
- Optionally sends Euler angles (in degrees or radians) derived from the IMU orientation quaternion.
- No data is logged or saved; this script is for real-time streaming only.

Usage:
    python sificonnect.py --ip 127.0.0.1 --port 16000 [--no-emg] [--no-imu]

OSC messages sent:
    /sifi/emg/{key}: value for each EMG channel
    /sifi/imu/{key}: value for each IMU channel
    /sifi/imu/roll, /sifi/imu/pitch, /sifi/imu/yaw: Euler angles (if enabled)
"""

import sifi_bridge_py as sbp
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
import argparse
import asyncio

# --- Config variables ---
RETURN_EULER = True  # Set to True to stream Euler angles, False to disable
EULER_ANGLES = "rad"  # Set to "deg" for degrees, "rad" for radians
STREAM_G = True      # Set to True to stream acceleration in g, False for m/s^2

# --- Utils function ---
def quart_to_eul(w, x, y, z, angle="rad"):
    """
    Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
    angle: "deg" for degrees, "rad" for radians.
    """
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    if angle == "deg":
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
    return roll, pitch, yaw

def m_s2_to_g(val):
    """Convert acceleration from m/s^2 to g."""
    return val / 9.80665

async def stream_sifi(sifi_ip="127.0.0.1", sifi_port=16000, EMG=True, IMU=True):
    """
    Connects to the SiFi band and streams data via OSC.

    Args:
        sifi_ip (str): OSC target IP address.
        sifi_port (int): OSC target port.
        EMG (bool): Whether to stream EMG data.
        IMU (bool): Whether to stream IMU data.
    """
    dispatcher = SimpleUDPClient(sifi_ip, sifi_port)
    sb = sbp.SifiBridge(data_transport="stdout")
    while not sb.connect():
        print("SiFi Connecting")
    print("SiFi Connected")
    print(f"SiFi Channels Set: {sb.set_channels(False, EMG, False, IMU, False)}")
    sb.set_filters(False)
    print(f"SiFi Configuration: {sb.show()}")
    sb.start()
    print(f"Streaming SiFi data to {sifi_ip}:{sifi_port}")
    try:
        while True:
            packet = sb.get_data()
            if packet["packet_type"] == "imu" and IMU:
                for key in packet["data"]:
                    for val in packet["data"][key]:
                        v = m_s2_to_g(val) if STREAM_G and "a" in key else val
                        dispatcher.send_message(f"/sifi/imu/{key}", v)
                if RETURN_EULER:
                    w_l, x_l, y_l, z_l = packet["data"]["qw"], packet["data"]["qx"], packet["data"]["qy"], packet["data"]["qz"]
                    for w, x, y, z in zip(w_l, x_l, y_l, z_l):
                        roll, pitch, yaw = quart_to_eul(w, x, y, z, angle=EULER_ANGLES)
                        dispatcher.send_message(f"/sifi/imu/roll", roll)
                        dispatcher.send_message(f"/sifi/imu/pitch", pitch)
                        dispatcher.send_message(f"/sifi/imu/yaw", yaw)
            if packet["packet_type"] == "emg_armband" and EMG:
                for key in packet["data"]:
                    for val in packet["data"][key]:
                        dispatcher.send_message(f"/sifi/emg/{key}", val)
            await asyncio.sleep(0)  # Yield control to event loop
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("Stopping SiFi client...")
        sb.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream SiFi band data via OSC.")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="OSC target IP address")
    parser.add_argument("--port", type=int, default=16000, help="OSC target port")
    parser.add_argument("--no-emg", action="store_true", help="Disable EMG streaming")
    parser.add_argument("--no-imu", action="store_true", help="Disable IMU streaming")
    args = parser.parse_args()
    asyncio.run(stream_sifi(sifi_ip=args.ip, sifi_port=args.port, EMG=not args.no_emg, IMU=not args.no_imu))


