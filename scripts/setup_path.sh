#!/bin/bash
# Copyright 2026 Sityu. All Rights Reserved.
#
# @file set_path.sh
# @brief Adds the directory containing this script to the PYTHONPATH variable.
#
# This script must be sourced rather than executed directly to modify the
# environment variables of the current shell session.

# Determine the absolute directory path of the script when sourced or executed.
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    script_path="${BASH_SOURCE[0]}"
else
    script_path="$0"
fi

# Resolve the physical absolute path to handle symbolic links correctly.
target_dir="$(cd -P "$(dirname "$script_path")" && pwd)"

# Append the target directory to PYTHONPATH if it is not already present.
if [[ ":$PYTHONPATH:" != *":$target_dir:"* ]]; then
    export PYTHONPATH="$target_dir:$PYTHONPATH"
    echo "Added to PYTHONPATH: $target_dir"
else
    echo "Already in PYTHONPATH: $target_dir"
fi
