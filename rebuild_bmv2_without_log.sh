#!/usr/bin/env bash
# rebuild.sh — Pulizia e ricompilazione pulita di BMv2 e p4c

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [-b BMV2_DIR] [-j NUM_CORES] [-h]

Options:
  -b BMV2_DIR   Path to BMv2 source (default: \$HOME/p4-tools/bmv2)
  -j NUM_CORES  Number of cores for make (default: all available)
  -h            Show this help message and exit
EOF
  exit 0
}

# Defaults
USER_HOME="$(eval echo ~${SUDO_USER-:-$USER})"
BMV2_DIR="${USER_HOME}/p4-tools/bmv2"
NUM_CORES="$(nproc)"

# Parse options
while getopts ":b:p:j:h" opt; do
  case "${opt}" in
    b) BMV2_DIR="${OPTARG}" ;;
    j) NUM_CORES="${OPTARG}" ;;
    h) usage ;;
    \?) echo "Invalid option: -${OPTARG}" >&2; usage ;;
    :)  echo "Option -${OPTARG} requires an argument." >&2; usage ;;
  esac
done

echo "=> BMv2 dir: $BMV2_DIR"
echo "=> Using $NUM_CORES cores for build"

# Rebuild BMv2
echo "=== Rebuilding BMv2 ==="
cd "$BMV2_DIR"
make clean || true
./autogen.sh
./configure --without-nanomsg --disable-elogger --disable-logging-macros --with-thrift \
            CFLAGS='-g -O2' CXXFLAGS='-g -O2'
make -j"$NUM_CORES"
sudo make install
sudo ldconfig

echo "=== Done: BMv2 e p4c ricompilati ==="
