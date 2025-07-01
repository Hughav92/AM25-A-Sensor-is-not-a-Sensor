"""
emg_test.py
-----------
Script for testing EMG data acquisition and processing.

Provides functions to acquire, process, and visualize EMG signals for testing purposes.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sifi_bridge_py as sbp
import keyboard 
import time

emg = []
time_l = []

sb = sbp.SifiBridge(data_transport="stdout")

while not sb.connect():
    print("SiFi Connecting")
print("SiFi Connected")

print(sb.set_channels(False, True, False, False, False)) # ECG EMG EDA IMU PPG
print(sb.set_filters(False))
sb.start()

previous_t = time.perf_counter()
s_t = time.perf_counter()
# Main loop with q key to exit
while True:
    # print(time.perf_counter()-s_t)
    # if keyboard.is_pressed('q'):  # Check if 'q' is pressed
    #     print("Exiting loop.")
    #     break  # Exit the loop
    print("hello")
    packet = sb.get_emg()
    print(packet)
    packet_t = time.perf_counter()
    time_l.append(packet_t - previous_t)
    previous_t = packet_t

    data = packet["data"]["emg0"]
    emg.extend(data)

write_time = time.time()
joblib.dump(time_l, f"test/time_l_{write_time}")
joblib.dump(emg, f"test/emg_{write_time}")

plt.plot(emg)