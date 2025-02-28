# Pytools-Lithography

## Introduction

This Jupyter Notebook was crafted as part of my Minor in Microtecnology, Processing and Devices (MPD) at The Hague University of Applied Sciences. While drawing inspiration from the work of previous students (Niek van Koolwijk and Lucas Sluitman), and the cross-platform compatibility updates by Emma Bajmat the code is further updated to use existing image analysis functions.

## Package Structure

The package is structured as follows:

```bash
pytools_lithography
├── analysis  # Module for all the analysis code
│   ├── legacy  # Old code that is not used anymore but still here for reference
│   ├── __init__.py
│   ├── analysis.py  # Main analysis functions
│   ├── batch_processing.py  # Functions to process multiple images at once
│   ├── image_processing.py  # Functions to process images
├── design  # Module for the design code (gdsfactory)
│   ├── __init__.py
│   ├── elektronics  # Module for elektronics design
│   ├── optics  # Module for optics design
│   ├── templates  # Module general templates
│   ├── routing.py  # Functions to route the design
│   ├── calibration.py  # Functions to calibrate the design
│   ├── tekst.py  # Functions to add text to the design
├── __init__.py
```

## Installation

### Prerequisites

The following software needs to be installed before installing the python package.

* git
* python (version above 3.10)

### Steps

The easiest way to use the code is by downloading the repo and using the example notebook.

```bash
# This code is for NON ANACONDA users on Linux
# Clone the repo
git clone https://github.com/Nynra/pytools-lithography.git
cd pytools-lithography

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate # Activate venv for linux
pip install --editable .  # Install the package
```

As I use VsCode and Linux myself I cannot provide instructions on how to use this code on Windows or Mac. All the code should work on Mac and Windows but the installation steps might be different and has not been tested. If you have any issues please let me know by creating an issue on the github page.

## Gotchas

* The code works but has not been tested a lot, make sure all the stripes
are horizontal and crop off the ends of the lines.
* Do NOT resize your images, only crop. Otherwise the nm/pixel factor calculated from the size bar will be wrong without giving a clear error.
* Make it easy on yourself and keep one path for all your analyses ... otherwise you will get a different path everywhere and suddenly you are editing three different images.
* If you are using this code for the minor make sure to cite the source repo to prevent a possible plagiarism flag.

## Sources

* Code developed by Niek van Koolwijk and Lucas Sluitman for the MPD HHS minor
* Code developed by Emma Bajmat for the MPD HHS minor
* [Paper on extracting litho parameters from SEM images](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9050/90500L/Determination-of-line-edge-roughness-in-low-dose-top-down/10.1117/12.2046493.short)

## Resources

* [GDSFactory Documentation](https://gdsfactory.readthedocs.io/en/latest/)
* [EasyOCR tutorial](https://medium.com/@adityamahajan.work/easyocr-a-comprehensive-guide-5ff1cb850168)
