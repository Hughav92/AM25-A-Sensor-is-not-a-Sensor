# A Sensor is not a Sensor: Diffracting the Preservation of Sonic Microinteraction with the SiFiBand

An accompanying repository to the paper *A Sensor is not a Sensor: Diffracting the Preservation of Sonic Microinteraction with the SiFiBand* presented at the *AudioMostly* conference 2025. This repository also serves as an archive for the SiFiBand version of *Stillness Under Tension*.

## Contents

- ```stilness_under_tension``` - contains the Myo and SiFiBand versions of *Stillness Under Tension*
- ```datalogger``` - contains a commandline tool for simulatneously logging Myo, SiFiBand, Qualisys Track Manager, and audio data
- ```analysis``` - contains Jupyter Notebooks with the analyses described in the paper
- ```paper``` - contains the paper presented at *AudioMostly* 2025.

## Installation

Create a new environment:

```bash
python3.10 -m venv stillness_under_tension
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For installing *Stillness Under Tension*, the following steps must be performed:

1. Download and install [Pure Data](https://puredata.info/downloads/pure-data).
2. For the SIFIBand version, download the [SiFiBridge.exe](https://github.com/SiFiLabs/sifi-bridge-pub?tab=readme-ov-file) and place it in the directory ```stillness_under_tension```.
3. For the Myo version, place the scripts:
    - ```__init__.py```
    - ```commands.py```
    - ```constants.py```
    - ```core.py```
    - ```profile.py```
    - ```types.py```
    - ```version.py```
  from the [dl-myo](https://github.com/iomz/dl-myo) repository in the directory ```stillness_under_tension/myo```.

For installing the datalogger:

1. Place the [dl-myo](https://github.com/iomz/dl-myo) scripts in the directory ```datalogger/myo```.
2. For communication with the *BitScope Micro*, place the scripts:
    - ```__init__.py```
    - ```analysis.py```
    - ```scope.py```
    - ```streams.py```
    - ```test.py```
    - ```utils.py```
    - ```vm.py```
    - ```VM registers.xlsx```
  from the [ScopeThing](https://github.com/jonathanhogg/scopething) repository in the directory ```datalogger/scopething```.

## Usage

To run *Stillness Under Tension:

1. Run either ```myoconnect.py``` or ```sificonnect.py```:

```bash
python myoconnect.py --ip 127.0.0.1 --port 17000 [--no-emg] [--no-imu]
```

```bash
python sificonnect.py --ip 127.0.0.1 --port 16000 [--no-emg] [--no-imu]
```

2. Run correspondingly either ```myo.pd``` or ```sifiband.pd```.

To run the datalogger:

```bash
python datalogger.py
```

Commandline instructions on usage are provided. Alternatively, baseline parameters can be set in ```datalogger/config.py```. 


## Requirements

See [requirements.txt](requirements.txt) for the full list of dependencies.

## License

CC-BY License
