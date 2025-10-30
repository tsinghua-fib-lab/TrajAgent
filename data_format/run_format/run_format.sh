#!/bin/bash

# Get the script name and path from the command-line arguments
SCRIPT_NAME=$1
SCRIPT_PATH=$2

# Check if both arguments are provided
if [ -z "$SCRIPT_NAME" ] || [ -z "$SCRIPT_PATH" ]; then
  echo "Error: Both script name and path are required"
  exit 1
fi

# Construct the full path to the Python script
FULL_SCRIPT_PATH="${SCRIPT_PATH}/${SCRIPT_NAME}"

# Check if the script exists
if [ ! -f "${FULL_SCRIPT_PATH}" ]; then
  echo "Error: Script not found"
  exit 1
fi

# Run the Python script using the full path
/usr/local/anaconda3/envs/torch-1.9.0-cu111-py38/bin/python "${FULL_SCRIPT_PATH}"