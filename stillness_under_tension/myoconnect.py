"""
myoconnect.py
-------------
Streams Myo armband EMG and IMU data via OSC (Open Sound Control) to a specified IP and port.

- Connects to a Myo device using the dongless-myo library. This can be found here: https://github.com/iomz/dl-myo
    The following scripts need to be placed in the myo directory:
    - __init__.py
    - commands.py
    - constants.py
    - core.py
    - profile.py
    - types.py
    - version.py
- Streams EMG and IMU data in real time.
- Optionally sends Euler angles (in degrees or radians) derived from the IMU orientation quaternion.
- No data is logged or saved; this script is for real-time streaming only.

Usage:
    python myoconnect.py --ip 127.0.0.1 --port 17000 [--no-emg] [--no-imu]

OSC messages sent:
    /myo/emg0 ... /myo/emg7: [sample1, sample2] for each EMG channel
    /myo/acceleration: [ax, ay, az]
    /myo/orientation: [qw, qx, qy, qz]
    /myo/gyroscope: [gx, gy, gz]
    /myo/euler: [roll, pitch, yaw] (if enabled)
"""

import asyncio
import numpy as np
from myo.core import MyoClient, EMGData, IMUData, EMGMode, IMUMode, ClassifierMode
from pythonosc.udp_client import SimpleUDPClient

# --- Config variables ---
RETURN_EULER = True  # Set to True to stream Euler angles, False to disable
EULER_ANGLES = "rad"  # Set to "deg" for degrees, "rad" for radians

# --- Utils function ---
def quart_to_eul(w, x, y, z, angle="rad"):
    """
    Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).

    Args:
        w, x, y, z (float): Quaternion components.
        angle (str): 'deg' for degrees, 'rad' for radians (default: 'rad').

    Returns:
        tuple: (roll, pitch, yaw) in specified units.
    """
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    # Only supports 'xyz' order for now
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    if angle == "deg":
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
    return roll, pitch, yaw

class MyoDataStreamer(MyoClient):
    """
    Handles streaming of Myo EMG and IMU data via OSC.
    Inherits from MyoClient and overrides data callbacks to send OSC messages.
    """
    def __init__(self, aggregate_all=False, aggregate_emg=False):
        """
        Initialize the MyoDataStreamer.

        Args:
            aggregate_all (bool): Aggregate all data (passed to MyoClient).
            aggregate_emg (bool): Aggregate EMG data (passed to MyoClient).
        """
        super().__init__(aggregate_all=aggregate_all, aggregate_emg=aggregate_emg)
        self.dispatch = False

    async def on_emg_data(self, emg: EMGData, timestamp):
        """
        Callback for incoming EMG data. Sends each channel's data via OSC.

        Args:
            emg (EMGData): EMG data object.
            timestamp (float): Timestamp of the data.
        """
        emg_dict = emg.to_dict()
        sample_1 = list(emg_dict["sample1"])
        sample_2 = list(emg_dict["sample2"])
        for i in range(8):
            if self.dispatch:
                self.dispatcher.send_message(f"/myo/emg{i}", [sample_1[i], sample_2[i]])

    async def on_imu_data(self, imu: IMUData, timestamp):
        """
        Callback for incoming IMU data. Sends acceleration, orientation, gyroscope, and optionally Euler angles via OSC.

        Args:
            imu (IMUData): IMU data object.
            timestamp (float): Timestamp of the data.
        """
        imu_dict = imu.to_dict()
        acceleration = list(imu_dict["accelerometer"])
        orientation = list(imu_dict["orientation"].values())
        gyroscope = list(imu_dict["gyroscope"])
        if RETURN_EULER:
            w, x, y, z = orientation
            roll, pitch, yaw = quart_to_eul(w, x, y, z, angle=EULER_ANGLES)
        if self.dispatch:
            self.dispatcher.send_message("/myo/acceleration", acceleration)
            self.dispatcher.send_message("/myo/orientation", orientation)
            self.dispatcher.send_message("/myo/gyroscope", gyroscope)
            if RETURN_EULER:
                self.dispatcher.send_message("/myo/euler", [roll, pitch, yaw])

    def dispatch_on(self):
        """
        Enable OSC data dispatching.
        """
        self.dispatch = True

    def dispatch_off(self):
        """
        Disable OSC data dispatching.
        """
        self.dispatch = False

    def set_up_dispatcher(self, myo_ip, myo_port):
        """
        Set up the OSC dispatcher with the given IP and port.

        Args:
            myo_ip (str): Target IP address.
            myo_port (int): Target port.
        """
        self.dispatcher = SimpleUDPClient(myo_ip, myo_port)

async def stream_myo(myo_ip="127.0.0.1", myo_port=17000, EMG=True, IMU=True):
    """
    Connects to the Myo device and streams data via OSC.

    Args:
        myo_ip (str): OSC target IP address.
        myo_port (int): OSC target port.
        EMG (bool): Whether to stream EMG data.
        IMU (bool): Whether to stream IMU data.
    """
    myo_client = await MyoDataStreamer.with_device(aggregate_all=False, aggregate_emg=False)
    myo_client.set_up_dispatcher(myo_ip, myo_port)
    myo_client.dispatch_on()
    await myo_client.setup(
        classifier_mode=ClassifierMode.DISABLED,
        emg_mode=EMGMode.SEND_RAW if EMG else EMGMode.NONE,
        imu_mode=IMUMode.SEND_DATA if IMU else IMUMode.NONE
    )
    print(f"Streaming Myo data to {myo_ip}:{myo_port}")
    await myo_client.start()
    try:
        while True:
            await asyncio.sleep(1)
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("Stopping Myo client...")
    await myo_client.stop()
    await myo_client.sleep()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stream Myo data via OSC.")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="OSC target IP address")
    parser.add_argument("--port", type=int, default=17000, help="OSC target port")
    parser.add_argument("--no-emg", action="store_true", help="Disable EMG streaming")
    parser.add_argument("--no-imu", action="store_true", help="Disable IMU streaming")
    args = parser.parse_args()
    asyncio.run(stream_myo(myo_ip=args.ip, myo_port=args.port, EMG=not args.no_emg, IMU=not args.no_imu))


