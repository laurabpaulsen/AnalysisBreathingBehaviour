#!/bin/bash

# Get the directory of the Bash script
scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

# Check if the virtual environment exists
if [ ! -d "$scriptDir/venv" ]; then
    echo "Virtual environment not found in $scriptDir/venv. Please create the virtual environment first."
    exit 1
fi

# Activate the virtual environment
source "$scriptDir/venv/bin/activate"

# Install IPython kernel within the virtual environment
python -m ipykernel install --user --name=venv