"""
generate_waveform.py
-------------------
Handles waveform generation for stimulation or testing.

Provides functions to generate and control various waveforms for use with the datalogger system.
"""

import sys
sys.path.append("./scopething")
from scope import await_, main, start_waveform, stop_waveform
import time
import asyncio
import queue

def generate_waveform(f, duration, initialisation_wait=0, waveform="sine", ratio=0.5, low=0, high=None, output_queue=None, task_queue=None):

    await_(main())
    
    kwargs = {
        "frequency": f,
        "waveform": waveform,
        "ratio": ratio,
        "low": low,
        "high": high
    }

    start_time = time.time()

    # add wait time to ensure not cutting off whole signal
    if initialisation_wait > 0:

        initialisation_start = time.perf_counter()
        while time.perf_counter()-initialisation_start < initialisation_wait:
            pass

    start_waveform(**kwargs)
    waveform_stopped = False

    counter_start = time.perf_counter()

    while time.perf_counter()-counter_start < duration:
        if task_queue is not None:
            try:
                task = task_queue.get_nowait()
                if task == "STOP":
                    stop_waveform()
                    waveform_stopped = True
                    break
            except queue.Empty:
                pass
        else:
            pass

    stop_time = time.time()

    if not waveform_stopped:
        stop_waveform()

    times = [start_time, stop_time]
    if output_queue is not None:
        output_queue.put(times)

if __name__ == "__main__":

    generate_waveform(10, 10)