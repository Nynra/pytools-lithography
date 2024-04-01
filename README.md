# Pytools-Lithography

## Introduction

This Jupyter Notebook was crafted as part of my Minor in Microtecnology, Processing and Devices (MPD) at The Hague University of Applied Sciences. While drawing inspiration from the work of previous students (Niek van Koolwijk and Lucas Sluitman), and the cross-platform compatibility updates by Emma Bajmat the code is further updated to use existing image analysis functions.

## Installation

The easiest way to use the code is by downloading the repo and using the example notebook.

```bash
# Clone the repo
git clone https://github.com/Nynra/pytools-lithography.git
cd pytools-lithography

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate # Activate venv for linux
pip install .  # Install the package
```

## Gotchas

- Make it easy on yourself and keep one path for all your analyses ... otherwise you will get a different path everywhere and suddenly you are editing three different images. 
- If you are using this code for the minor make sure to cite the source repo to prevent a possible plagiarism flag.

## Sources

- Code developed by Niek van Koolwijk and Lucas Sluitman for the MPD HHS minor
- Code developed by Emma Bajmat for the MPD HHS minor
- (Paper on extracting litho parameters from SEM images)[https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9050/90500L/Determination-of-line-edge-roughness-in-low-dose-top-down/10.1117/12.2046493.short]
