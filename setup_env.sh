#!/bin/bash

# Get the directory of the Bash script
scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

# Create a virtual environment in the same directory as the script
python3 -m venv "$scriptDir/venv"

# Activate the virtual environment
source "$scriptDir/venv/bin/activate"
pip install --upgrade pip

# Install requirements from the same directory as the script
python3 -m pip install -r "$scriptDir/requirements.txt"


# local pyriodic package:
pip install /Users/au661930/Library/CloudStorage/OneDrive-Aarhusuniversitet/Dokumenter/projects/_BehaviouralBreathing/pyriodic


# make it jupyter kernel

python3 -m ipykernel install --user --name=BreathingBehaviourAnalysis

echo "Done!"