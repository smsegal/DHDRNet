#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

all_paths="$(fd -e="dng" . './data/HDR+/20171106/results_20171023/' |
	awk -F/ '{print "./" $0 " ./" $1 "/" $2 "/dngs/" $(NF-1) ".dng"}')"

for cpath in $all_paths; do
    IFS=' ' read -r -a array <<< "$cpath"
    # echo source "${array[0]}"
    # echo sink "${array[1]}"
    ln -s $(realpath "${array[0]}") "${array[1]}"
done
