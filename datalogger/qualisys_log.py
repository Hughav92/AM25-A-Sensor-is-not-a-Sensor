"""
qualisys_log.py
--------------
Handles Qualisys motion capture data logging and streaming for the datalogger system.

Provides functions to start, stop, and clean up Qualisys data acquisition processes.
"""

import datetime
import os
import asyncio
import time
import joblib
import pandas as pd
from pythonosc.udp_client import SimpleUDPClient
from osc4py3.as_eventloop import osc_startup, osc_udp_server, osc_process, osc_method, osc_terminate
from osc4py3 import oscmethod as osm
from config import rigid_body_addresses, command_ip, command_port, server_ip, server_port
from multiprocessing import Event

buffer_dict = {}
dump_count = 0
command_client = None

def osc_to_buffer_handler_factory(buffer_dict):
    """
    Factory function to create an OSC handler that logs data to a buffer.
    
    Args:
        buffer_dict (dict): The buffer dictionary to store data.
    
    Returns:
        function: The OSC handler function.
    """
    def osc_to_buffer_handler(address, *args):
        address = address.split("/")[-1]
        buffer_dict[address]["pos_x"].append(args[0])
        buffer_dict[address]["pos_y"].append(args[1])
        buffer_dict[address]["pos_z"].append(args[2])
        buffer_dict[address]["rot_0"].append(args[3])
        buffer_dict[address]["rot_1"].append(args[4])
        buffer_dict[address]["rot_2"].append(args[5])
        buffer_dict[address]["rot_3"].append(args[6])
        buffer_dict[address]["rot_4"].append(args[7])
        buffer_dict[address]["rot_5"].append(args[8])
        buffer_dict[address]["rot_6"].append(args[9])
        buffer_dict[address]["rot_7"].append(args[10])
        buffer_dict[address]["rot_8"].append(args[11])
        buffer_dict[address]["receive_timestamp"].append(time.time())
    return osc_to_buffer_handler

def setup_osc_server(ip, port, osc_addresses, buffer_dict):
    """
    Sets up the OSC server to receive data.
    
    Args:
        ip (str): The IP address of the server.
        port (int): The port of the server.
        osc_addresses (list): The list of OSC addresses to listen to.
        buffer_dict (dict): The buffer dictionary to store data.
    """
    osc_startup()
    osc_udp_server(ip, port, "server")
    log_handler = osc_to_buffer_handler_factory(buffer_dict)
    osc_method("/qtm/cmd_res", osc_command_print_handler, argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_DATAUNPACK)
    for address in osc_addresses:
        osc_method(f"/qtm/6d/{address}", log_handler, argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_DATAUNPACK)

def osc_command_print_handler(address, *args):
    """
    Handler function to print OSC command responses.
    
    Args:
        address (str): The OSC address.
        *args: The arguments of the OSC message.
    """
    print(address, args)

async def process_osc():
    """
    Processes OSC messages in an asynchronous loop.
    """
    while True:
        osc_process()
        await asyncio.sleep(0)

async def qualisys_log(log_name=None, stop_event=None):
    """
    Logs data from the Qualisys system.
    
    Args:
        log_name (str): The name of the log file.
    """
    global buffer_dict, dump_count, command_client

    if log_name is None:
        log_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    log_path = os.path.join("logs", log_name, "qualisys")
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
        for address in rigid_body_addresses:
            os.makedirs(os.path.join(log_path, address), exist_ok=True)
        print(f"Folder created at {log_path}")
    else:
        print(f"Folder already exists at {log_path}")

    buffer_dict = {address: {
        "pos_x": [], "pos_y": [], "pos_z": [],
        "rot_0": [], "rot_1": [], "rot_2": [], "rot_3": [],
        "rot_4": [], "rot_5": [], "rot_6": [], "rot_7": [],
        "rot_8": [], "receive_timestamp": []} for address in rigid_body_addresses}

    setup_osc_server(server_ip, server_port, rigid_body_addresses, buffer_dict)

    command_client = SimpleUDPClient(command_ip, command_port)
    osc_server_task = asyncio.create_task(process_osc())

    command_client.send_message("/qtm", "Connect 45455")
    command_client.send_message("/qtm", "StreamFrames AllFrames 6D")

    stop_triggered = False

    try:
        counter = 0
        dump_count = 0
        stop_triggered = False
        while True:

            counter += 1

            if counter % 100000 == 0:
                for address in rigid_body_addresses:
                    file_path = os.path.join("logs", log_name, "qualisys", address, f"qualisys_{address}_dict_{str(dump_count).zfill(5)}.joblib")
                    joblib.dump(buffer_dict[address], file_path)
                    buffer_dict[address] = {
                        "pos_x": [], "pos_y": [], "pos_z": [],
                        "rot_0": [], "rot_1": [], "rot_2": [], "rot_3": [],
                        "rot_4": [], "rot_5": [], "rot_6": [], "rot_7": [],
                        "rot_8": [], "receive_timestamp": []}
                dump_count += 1

            if stop_event.is_set() and not stop_triggered:
                stop_triggered = True
                if command_client:
                    command_client.send_message("/qtm", "Disconnect")
                    print("Qualisys Connection Shut")

                osc_terminate()
                print("OSC server terminated.")
                
                if buffer_dict:
                    for address in rigid_body_addresses:
                        file_path = os.path.join("logs", log_name, "qualisys", address, f"qualisys_{address}_dict_{str(dump_count).zfill(5)}.joblib")
                        if buffer_dict[address]:
                            joblib.dump(buffer_dict[address], file_path)

            await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print("Process interrupted by user.")
        qualisys_cleanup(log_name)

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        print("Shutting down...")
        qualisys_cleanup(log_name)

def qualisys_cleanup(log_name):
    """
    Cleans up and saves the Qualisys data.
    
    Args:
        log_name (str): The name of the log file.
    """

    for address in rigid_body_addresses:
        address_output = {}
        files = sorted(os.listdir(os.path.join("logs", log_name, "qualisys", address)))
        for file in files:
            if file.endswith('.joblib'):
                file_path = os.path.join("logs", log_name, "qualisys", address, file)
                data = joblib.load(file_path)
                for key, value in data.items():
                    if key not in address_output:
                        address_output[key] = []
                    address_output[key].extend(value)

        address_df = pd.DataFrame(address_output)
        csv_path = os.path.join("logs", log_name, "qualisys", address, f"qualisys_{address}.csv")
        address_df.to_csv(csv_path, index=False)
        print(f"CSV written for {address}")

def qualisys_main(log_name=None, stop_event=None):
    """
    Main function to start logging Qualisys data.
    
    Args:
        log_name (str): The name of the log file.
    """
    try:
        asyncio.run(qualisys_log(log_name, stop_event))
    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Final cleanup...")
