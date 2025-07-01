"""
myo_log.py
----------
Handles Myo armband data logging and streaming for the datalogger system.

Provides functions to start, stop, and clean up Myo data acquisition processes.
"""

import asyncio
import os
import numpy as np
import matplotlib.pyplot as plt
import collections
from datetime import datetime
from myo.core import MyoClient, EMGData, IMUData, EMGMode, IMUMode, ClassifierMode
import joblib
import pandas as pd
import itertools
import sys
sys.path.append('../utils/')
from utils import quart_to_eul
from config import RETURN_EULER, EULER_ANGLES
from pythonosc.udp_client import SimpleUDPClient
from multiprocessing import Event

def prepend_start_time(df, start_time):
    """
    Prepends the start time to the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to prepend the start time to.
        start_time (float): The start time to prepend.
    
    Returns:
        pd.DataFrame: The DataFrame with the start time prepended.
    """
    empty_row = {col: "" for col in df.columns}
    if "receive_timestamp" in df.columns:
        empty_row["receive_timestamp"] = start_time
    df = pd.concat([pd.DataFrame([empty_row]), df], ignore_index=True)
    return df

# Global variables for storing data
start_time = None
emg_buffer = [[] for _ in range(8)]
emg_timestamps = []
acceleration_buffer = [[] for _ in range(3)]
orientation_buffer = [[] for _ in range(4)]
gyroscope_buffer = [[] for _ in range(3)]
imu_timestamps = []
dump_count = 0

class MyoDataStreamer(MyoClient):
    """
    A class to handle streaming data from the Myo device.
    """

    def __init__(self, aggregate_all=False, aggregate_emg=False):
        super().__init__(aggregate_all, aggregate_emg)
        self.dispatch = False

    async def on_emg_data(self, emg: EMGData, timestamp):
        """
        Callback function for handling EMG data.
        
        Args:
            emg (EMGData): The EMG data.
            timestamp (float): The timestamp of the data.
        """
        emg_dict = emg.to_dict()
        sample_1 = list(emg_dict["sample1"])
        sample_2 = list(emg_dict["sample2"])

        for i in range(8):
            emg_buffer[i].append(sample_1[i])
            emg_buffer[i].append(sample_2[i])
            
            if self.dispatch:
                self.dispatcher.send_message(f"/myo/emg{i}", [sample_1[i], sample_2[i]])

        for _ in range(2):
            emg_timestamps.append(timestamp)

    async def on_imu_data(self, imu: IMUData, timestamp):
        """
        Callback function for handling IMU data.
        
        Args:
            imu (IMUData): The IMU data.
            timestamp (float): The timestamp of the data.
        """
        imu_dict = imu.to_dict()
        acceleration = list(imu_dict["accelerometer"])
        orientation = list(imu_dict["orientation"].values())
        gyroscope = list(imu_dict["gyroscope"])

        for i in range(3):
            acceleration_buffer[i].append(acceleration[i])
            gyroscope_buffer[i].append(gyroscope[i])

        for i in range(4):
            orientation_buffer[i].append(orientation[i])

        if RETURN_EULER:
            w, x, y, z = orientation
            roll, pitch, yaw = quart_to_eul(w, x, y, z, angle=EULER_ANGLES)

        if self.dispatch:
            self.dispatcher.send_message("/myo/acceleration", acceleration)
            self.dispatcher.send_message("/myo/orientation", orientation)
            self.dispatcher.send_message("/myo/gyroscope", gyroscope)
            if RETURN_EULER:
                self.dispatcher.send_message("/myo/euler", [roll, pitch, yaw])

        imu_timestamps.append(timestamp)

    def dispatch_on(self):
        """
        Enables data dispatching.
        """

        self.dispatch = True

    def dispatch_off(self):
        """
        Disables data dispatching.
        """
        self.dispatch = False

    def set_up_dispatcher(self, myo_ip, myo_port):
        """
        Sets up the dispatcher for sending data.
        
        Args:
            myo_ip (str): The IP address of the dispatcher.
            myo_port (int): The port of the dispatcher.
        """
        self.dispatcher = SimpleUDPClient(myo_ip, myo_port)

async def log_myo(log_name, EMG=True, IMU=True, myo_stream=False, myo_ip="127.0.0.1", myo_port=17000, stop_event=None):
    """
    Logs data from the Myo device.
    
    Args:
        log_name (str): The name of the log file.
        EMG (bool): Whether to log EMG data.
        IMU (bool): Whether to log IMU data.
        myo_stream (bool): Whether to stream data.
        myo_ip (str): The IP address for streaming.
        myo_port (int): The port for streaming.
    """
    global emg_buffer, emg_timestamps, acceleration_buffer, orientation_buffer, gyroscope_buffer, imu_timestamps, start_time, dump_count
      
    myo_client = await MyoDataStreamer.with_device(aggregate_all=False, aggregate_emg=False)

    print(myo_stream)

    if myo_stream:
        
        myo_client.set_up_dispatcher(myo_ip, myo_port)
        myo_client.dispatch_on()
    
    if myo_client:
        await myo_client.setup(
            classifier_mode=ClassifierMode.DISABLED,
            emg_mode=EMGMode.SEND_RAW if EMG else EMGMode.NONE,
            imu_mode=IMUMode.SEND_DATA if IMU else IMUMode.NONE
        )
        print("Connecting to Myo")
        await myo_client.start()

        start_time = myo_client.subscription_start_time
        joblib.dump(start_time, os.path.join("logs", log_name, "myo", "start_time.joblib"))
        count = 0
        dump_count = 0
        try:
            stop_triggered = False

            while True:
                count += 1
                if count % 10 == 0:
                    joblib.dump(emg_buffer, os.path.join("logs", log_name, "myo", "emg", f"emg_data_{str(dump_count).zfill(5)}.joblib"))
                    emg_buffer = [[] for _ in range(8)]
                    joblib.dump(emg_timestamps, os.path.join("logs", log_name, "myo", "emg", f"emg_timestamps_{str(dump_count).zfill(5)}.joblib"))
                    emg_timestamps = []

                    joblib.dump(acceleration_buffer, os.path.join("logs", log_name, "myo", "imu", f"acceleration_{str(dump_count).zfill(5)}.joblib"))
                    acceleration_buffer = [[] for _ in range(3)]
                    joblib.dump(orientation_buffer, os.path.join("logs", log_name, "myo", "imu", f"orientation_{str(dump_count).zfill(5)}.joblib"))
                    orientation_buffer = [[] for _ in range(4)]
                    joblib.dump(gyroscope_buffer, os.path.join("logs", log_name, "myo", "imu", f"gyroscope_{str(dump_count).zfill(5)}.joblib"))
                    gyroscope_buffer = [[] for _ in range(3)]
                    joblib.dump(imu_timestamps, os.path.join("logs", log_name, "myo", "imu", f"imu_timestamps_{str(dump_count).zfill(5)}.joblib"))
                    imu_timestamps = []

                    dump_count +=1

                await asyncio.sleep(1)  # Keeps the event loop running

                if stop_event.is_set() and not stop_triggered:
                    stop_triggered = True
                    if emg_buffer:
                        joblib.dump(emg_buffer, os.path.join("logs", log_name, "myo", "emg", f"emg_data_{str(dump_count).zfill(5)}.joblib"))
                    if emg_timestamps:
                        joblib.dump(emg_timestamps, os.path.join("logs", log_name, "myo", "emg", f"emg_timestamps_{str(dump_count).zfill(5)}.joblib"))
                    if acceleration_buffer:
                        joblib.dump(acceleration_buffer, os.path.join("logs", log_name, "myo", "imu", f"acceleration_{str(dump_count).zfill(5)}.joblib"))
                    if orientation_buffer:
                        joblib.dump(orientation_buffer, os.path.join("logs", log_name, "myo", "imu", f"orientation_{str(dump_count).zfill(5)}.joblib"))
                    if gyroscope_buffer:
                        joblib.dump(gyroscope_buffer, os.path.join("logs", log_name, "myo", "imu", f"gyroscope_{str(dump_count).zfill(5)}.joblib"))
                    if imu_timestamps:
                        joblib.dump(imu_timestamps, os.path.join("logs", log_name, "myo", "imu", f"imu_timestamps_{str(dump_count).zfill(5)}.joblib"))
                    print("Stopping Myo client...")

        except asyncio.CancelledError:
            joblib.dump(emg_buffer, os.path.join("logs", log_name, "myo", "emg", f"emg_data_{str(dump_count).zfill(5)}.joblib"))
            joblib.dump(emg_timestamps, os.path.join("logs", log_name, "myo", "emg", f"emg_timestamps_{str(dump_count).zfill(5)}.joblib"))
            joblib.dump(acceleration_buffer, os.path.join("logs", log_name, "myo", "imu", f"acceleration_{str(dump_count).zfill(5)}.joblib"))
            joblib.dump(orientation_buffer, os.path.join("logs", log_name, "myo", "imu", f"orientation_{str(dump_count).zfill(5)}.joblib"))
            joblib.dump(gyroscope_buffer, os.path.join("logs", log_name, "myo", "imu", f"gyroscope_{str(dump_count).zfill(5)}.joblib"))
            joblib.dump(imu_timestamps, os.path.join("logs", log_name, "myo", "imu", f"imu_timestamps_{str(dump_count).zfill(5)}.joblib"))
            print("Stopping Myo client...")

        await myo_client.stop()
        await myo_client.sleep()

def myo_main(log_name=None, EMG=True, IMU=True, myo_stream=False, myo_ip="127.0.0.1", myo_port=17000, stop_event=None):
    """
    Main function to start logging Myo data.
    
    Args:
        log_name (str): The name of the log file.
        EMG (bool): Whether to log EMG data.
        IMU (bool): Whether to log IMU data.
        myo_stream (bool): Whether to stream data.
        myo_ip (str): The IP address for streaming.
        myo_port (int): The port for streaming.
    """
    if log_name is None:
        log_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if not os.path.exists(os.path.join("logs", log_name, "myo")):
        if IMU:
            os.makedirs(os.path.join("logs", log_name, "myo", "imu"))
        if EMG:
            os.makedirs(os.path.join("logs", log_name, "myo", "emg"))
        print(f"Folder created at {os.path.join('logs', log_name, 'myo')}")
    else:
        print(f"Folder already exists at {os.path.join('logs', log_name, 'myo')}")

    try:
        asyncio.run(log_myo(log_name, EMG=EMG, IMU=IMU, myo_stream=myo_stream, myo_ip=myo_ip, myo_port=myo_port, stop_event=stop_event))
    except KeyboardInterrupt:
        myo_cleanup(log_name)

def myo_cleanup(log_name):
    """
    Cleans up and saves the Myo data.
    
    Args:
        log_name (str): The name of the log file.
    """

    start_time = joblib.load(os.path.join("logs", log_name, "myo", "start_time.joblib"))
    
    imu_output = {
        "ax": [],
        "ay": [],
        "az": [],
        "qw": [],
        "qx": [],
        "qy": [],
        "qz": [],
        "gx": [],
        "gy": [],
        "gz": [],
        "receive_timestamp": []
    }

    emg_output = {f"emg{i}": [] for i in range(8)}
    emg_output["receive_timestamp"] = []

    for file in sorted(os.listdir(os.path.join("logs", log_name, "myo", "imu"))):
        loaded_list = joblib.load(os.path.join("logs", log_name, "myo", "imu", file))

        if "acceleration" in file:
            imu_output["ax"].append(loaded_list[0])
            imu_output["ay"].append(loaded_list[1])
            imu_output["az"].append(loaded_list[2])

        if "orientation" in file:
            imu_output["qw"].append(loaded_list[0])
            imu_output["qx"].append(loaded_list[1])
            imu_output["qy"].append(loaded_list[2])
            imu_output["qz"].append(loaded_list[3])

        if "gyroscope" in file:
            imu_output["gx"].append(loaded_list[0])
            imu_output["gy"].append(loaded_list[1])
            imu_output["gz"].append(loaded_list[2])

        if "timestamps" in file:
            imu_output["receive_timestamp"].append(loaded_list)

    for file in sorted(os.listdir(os.path.join("logs", log_name, "myo", "emg"))):
        loaded_list = joblib.load(os.path.join("logs", log_name, "myo", "emg", file))

        if "emg_data" in file:
            for i in range(8):
                emg_output[f"emg{i}"].append(loaded_list[i])

        if "timestamps" in file:
            emg_output["receive_timestamp"].append(loaded_list)

    for key in imu_output:
        imu_output[key] = list(itertools.chain(*imu_output[key]))

    for key in emg_output:
        emg_output[key] = list(itertools.chain(*emg_output[key]))

    imu_df = pd.DataFrame(imu_output)
    emg_df = pd.DataFrame(emg_output)

    imu_df = prepend_start_time(imu_df, start_time)
    emg_df = prepend_start_time(emg_df, start_time)

    imu_df.to_csv(os.path.join("logs", log_name, "myo", "imu", "imu.csv"), index=False)
    emg_df.to_csv(os.path.join("logs", log_name, "myo", "emg", "emg.csv"), index=False)


