#!/usr/bin/env bash
# Apply the fusion patch series on top of the upstream PrismML fork.
#
#   ./apply-fusion.sh <workdir> [prism-commit]
#
# default base commit: 9a9394a (PrismML prism branch, the commit this series was
# cut against).  The result is the tree the shipped runtime was built from.
set -euo pipefail

WORK="${1:?usage: apply-fusion.sh <workdir> [base-commit]}"
BASE="${2:-9a9394a895b96003ca842a6041cb28ac49a108f7}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$WORK/.git" ]; then
    echo "[1/4] cloning PrismML-Eng/llama.cpp into $WORK"
    git clone --filter=blob:limit=204800 https://github.com/PrismML-Eng/llama.cpp.git "$WORK"
fi

cd "$WORK"
echo "[2/4] fetching + checking out base $BASE"
git fetch --no-tags origin prism
git checkout -B fusion-local "$BASE"

echo "[3/4] applying $(ls "$HERE"/*.patch | wc -l) patches"
git am --3way "$HERE"/*.patch

echo "[4/4] done"
cat <<'EOF'

Build it with the HIP toolchain (see docs/HARDWARE-GFX1030.md):

    cmake -S . -B build-hip -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DAMDGPU_TARGETS=gfx1030 \
      -DGGML_NATIVE=OFF \
      -DCMAKE_C_COMPILER="<rocm>/bin/clang.exe" \
      -DCMAKE_CXX_COMPILER="<rocm>/bin/clang++.exe" \
      -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT="$PWD"
    cmake --build build-hip --target llama-server llama-kvmem-server llama-kvmem-cli -j
EOF
