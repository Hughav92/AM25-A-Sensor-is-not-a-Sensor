"""
sifiband_log.py
---------------
Handles SiFi band data logging and streaming for the datalogger system.

Provides functions to start, stop, and clean up SiFi band data acquisition processes.
"""

import sifi_bridge_py as sbp
import time
import joblib
import os
from datetime import datetime
import pandas as pd
import numpy as np
from config import IMU, EMG, RETURN_EULER, EULER_ANGLES, RETURN_ANGULAR_VELOCITY, STREAM_G, sifi_imu_sr
import sys
sys.path.append('../utils/')
from utils import quart_to_eul, quart_to_ang_vel, m_s2_to_g
from pythonosc.udp_client import SimpleUDPClient
from multiprocessing import Event

start_time = 0
imu_dict = {}
emg_dict = {}
dump_count = 0

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

def log_sifiband(log_name=None, EMG=True, IMU=True, sifi_stream=False, sifi_ip="127.0.0.1", sifi_port=16000, stop_event=None):
    """
    Logs data from the SiFi band.
    
    Args:
        log_name (str): The name of the log file.
        EMG (bool): Whether to log EMG data.
        IMU (bool): Whether to log IMU data.
        sifi_stream (bool): Whether to stream data.
        sifi_ip (str): The IP address for streaming.
        sifi_port (int): The port for streaming.
    """
    global start_time, imu_dict, emg_dict, dump_count, RETURN_EULER, EULER_ANGLES, RETURN_ANGULAR_VELOCITY, sifi_imu_sr#, packet_log

    print(f"SiFi Logging EMG -> {EMG}, IMU -> {IMU}")

    if log_name is None:
        log_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if not os.path.exists(os.path.join("logs", log_name, "sifiband")):
        if IMU:
            os.makedirs(os.path.join("logs", log_name, "sifiband", "imu"))
        if EMG:
            os.makedirs(os.path.join("logs", log_name, "sifiband", "emg"))
        os.makedirs(os.path.join("logs", log_name, "sifiband", "logs"))
        print(f"Folder created at {os.path.join('logs', log_name, 'sifiband')}")
    else:
        print(f"Folder already exists at {os.path.join('logs', log_name, 'sifiband')}")

    if sifi_stream:
        dispatcher = SimpleUDPClient(sifi_ip, sifi_port)

    sb = sbp.SifiBridge(data_transport="stdout")

    while not sb.connect():
        print("SiFi Connecting")
    print("SiFi Connected")

    print(f"SiFi Channels Set: {sb.set_channels(False, EMG, False, IMU, False)}") # ECG EMG EDA IMU PPG
    sb.set_filters(False)
    print(f"SiFi Configuration: {sb.show()}")
    sb.start()
    start_time = time.time()
    joblib.dump(start_time, os.path.join("logs", log_name, "sifiband", "start_time.joblib"))
    
    print("Initiating SiFi Logs")
    imu_dict = {}
    emg_dict = {}

    imu_initiated = False
    emg_initiated = False

    # Initialize dictionaries for IMU and EMG data
    while not (imu_initiated and emg_initiated):
        if IMU == False:
            imu_initiated = True
        if EMG == False:
            emg_initiated = True
        packet = sb.get_data()
        if packet["packet_type"] == "imu":
            for key in packet["data"]:
                imu_dict[key] = []
                if RETURN_EULER:
                    imu_dict["roll"] = []
                    imu_dict["pitch"] = []
                    imu_dict["yaw"] = []
                imu_dict["receive_timestamp"] = []
            imu_initiated = True
        if packet["packet_type"] == "emg_armband":
            for key in packet["data"]:
                emg_dict[key] = []
                emg_dict["receive_timestamp"] = []
            emg_initiated = True

    print("SiFi logs initiated")

    packets_received = 0
    dump_count = 0

    try:

        stop_triggered = False

        if RETURN_ANGULAR_VELOCITY:
            previous_quats = np.array([0, 0, 0, 1])
        
        while True:
            packet = sb.get_data()
            receive_timestamp = time.time()
            if packet["packet_type"] == "imu":
                for key in packet["data"]:
                    data = packet["data"][key]
                    for val in data:
                        imu_dict[key].append(val)
                        if STREAM_G:
                            if "a" in key:
                                val = m_s2_to_g(val)
                        if sifi_stream:
                            dispatcher.send_message(f"/sifi/imu/{key}", val)
                if RETURN_EULER:
                    w_l, x_l, y_l, z_l = packet["data"]["qw"], packet["data"]["qx"], packet["data"]["qy"], packet["data"]["qz"]
                    for w, x, y, z in zip(w_l, x_l, y_l, z_l):
                        roll, pitch, yaw = quart_to_eul(w, x, y, z, angle=EULER_ANGLES)
                        imu_dict["roll"].append(roll)
                        imu_dict["pitch"].append(pitch)
                        imu_dict["yaw"].append(yaw)
                        dispatcher.send_message(f"/sifi/imu/roll", roll)
                        dispatcher.send_message(f"/sifi/imu/pitch", pitch)
                        dispatcher.send_message(f"/sifi/imu/yaw", yaw)
                if RETURN_ANGULAR_VELOCITY:
                    w_l, x_l, y_l, z_l = packet["data"]["qw"], packet["data"]["qx"], packet["data"]["qy"], packet["data"]["qz"]
                    for i, (x, y, z, w) in enumerate(zip(x_l, y_l, z_l, w_l)):
                        if i == 0:
                            q1 = previous_quats
                            q2 = np.array([x, y, z, w])
                            omega = quart_to_ang_vel(q1, q2, sifi_imu_sr)
                            dispatcher.send_message(f"/sifi/imu/omega", omega)
                        else:
                            q1 = np.array([x_l[i-1], y_l[i-1], z_l[i-1], w_l[i-1]])
                            q2 = np.array([x, y, z, w])
                            omega = quart_to_ang_vel(q1, q2, sifi_imu_sr)
                            dispatcher.send_message(f"/sifi/imu/omega", omega)
                    previous_quats = np.array([x_l[-1], y_l[-1], z_l[-1], w_l[-1]])
                for i in range(len(packet["data"][key])):
                    imu_dict["receive_timestamp"].append(receive_timestamp)
            if packet["packet_type"] == "emg_armband":
                for key in packet["data"]:
                    data = packet["data"][key]
                    for val in data:
                        emg_dict[key].append(val)
                        if sifi_stream:
                            dispatcher.send_message(f"/sifi/emg/{key}", val)
                for i in range(len(packet["data"][key])):
                    emg_dict["receive_timestamp"].append(receive_timestamp)
            packets_received += 1
            if packets_received % 500 == 0:
                if IMU:
                    joblib.dump(imu_dict, os.path.join("logs", log_name, "sifiband", "imu", f"imu_dict_{str(dump_count).zfill(5)}.joblib"))
                    for key in imu_dict.keys():
                        imu_dict[key] = []
                if EMG:
                    joblib.dump(emg_dict, os.path.join("logs", log_name, "sifiband", "emg", f"emg_dict_{str(dump_count).zfill(5)}.joblib"))
                    for key in emg_dict.keys():
                        emg_dict[key] = []
                dump_count += 1

            if stop_event.is_set() and not stop_triggered:

                stop_triggered = True
                if IMU:
                    joblib.dump(imu_dict, os.path.join("logs", log_name, "sifiband", "imu", f"imu_dict_{str(dump_count).zfill(5)}.joblib"))
                if EMG:
                    joblib.dump(emg_dict, os.path.join("logs", log_name, "sifiband", "emg", f"emg_dict_{str(dump_count).zfill(5)}.joblib"))

    except KeyboardInterrupt:
        print("Exiting SiFi")
        sifiband_cleanup(log_name, IMU, EMG)

def sifiband_cleanup(log_name, IMU, EMG):
    """
    Cleans up and saves the SiFi band data.
    
    Args:
        log_name (str): The name of the log file.
    """
    
    start_time = joblib.load(os.path.join("logs", log_name, "sifiband", "start_time.joblib"))
    if IMU:
        imu_output = {}
        for file in sorted(os.listdir(os.path.join("logs", log_name, "sifiband", "imu"))):
            dict = joblib.load(os.path.join("logs", log_name, "sifiband", "imu", file))
            for key, value in dict.items():
                if key not in imu_output:
                    imu_output[key] = []
                imu_output[key].extend(value)
        imu_df = pd.DataFrame(imu_output)
        imu_df = prepend_start_time(imu_df, start_time)
        imu_df.to_csv(os.path.join("logs", log_name, "sifiband", "imu", "imu.csv"), index=False)
    if EMG:
        emg_output = {}
        for file in sorted(os.listdir(os.path.join("logs", log_name, "sifiband", "emg"))):
            print(file)
            print(os.path.join("logs", log_name, "sifiband", "emg", file))
            dict = joblib.load(os.path.join("logs", log_name, "sifiband", "emg", file))
            for key, value in dict.items():
                if key not in emg_output:
                    emg_output[key] = []
                emg_output[key].extend(value)
        emg_df = pd.DataFrame(emg_output)
        emg_df = prepend_start_time(emg_df, start_time)
        emg_df.to_csv(os.path.join("logs", log_name, "sifiband", "emg", "emg.csv"), index=False)