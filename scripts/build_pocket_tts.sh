#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PTTS_PATH="${PROJECT_ROOT}/deps/PocketTTS.cpp"
MODELS="${PROJECT_ROOT}/models/TTS"

build_ptts() {
    BUILD='True'

    # Check if it is already built
    if [[ -d "${MODELS}" && -d "${PTTS_PATH}/.build" ]]; then
        echo "Already built PocketTTS.cpp...skipping"
        BUILD='False'
    fi

    # Force build override
    if [[ "$1" == "-f" ]]; then
        echo "Force build override"
        BUILD='True'
    fi

    if [[ "${BUILD}" == "True" ]]; then
        echo "Building PocketTTS.cpp..."
        pushd "${PTTS_PATH}"
        cmake -B .build -DCMAKE_BUILD_TYPE=Release
        cmake --build .build -j$(nproc)
        cd "${PROJECT_ROOT}"
        uv run python deps/PocketTTS.cpp/export_onnx.py --output-dir "${MODELS}"
        popd
    fi
}

if [ ! -d "${PTTS_PATH}" ]; then
    echo "Error: can't find PocketTTS.cpp"
    exit 1
fi

# Build deps
build_ptts $1

