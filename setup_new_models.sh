#!/bin/bash
# setup_new_models.sh
# One-time setup for the four new methods: dofnet / sida / aeroblade / safe.
# Creates symlinks (or empty markers) for `pytorch_model.bin` in each model dir
# so that start.sh's `scan_models` discovers them. Actual weights are loaded
# from paths in each model's config.json (or from elsewhere on disk).

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MD="${ROOT}/OhMyYuwan/face-forgery-detection"

# === Edit these paths if your weights live elsewhere ===
DOFNET_CKPT="/zyy/TIFS2026/Visualization/dofnet/DoFNet/ckpt/model_ckpt_12_16_50_0.4_0.3_0.3.pth"
SAFE_CKPT="/zyy/TIFS2026/Visualization/safe/checkpoint-best.pth"
SIDA_DIR="/zyy/TIFS2026/models/SIDA-13B"
AERO_SD1="/zyy/TIFS2026/Visualization/testfree/sd1"
AERO_SD2="/zyy/TIFS2026/Visualization/testfree/sd2_vae"
AERO_KD21="/zyy/TIFS2026/Visualization/testfree/kd21"

ok() { echo "  ✅ $1"; }
warn() { echo "  ⚠️  $1"; }
err() { echo "  ❌ $1"; }

echo "Setting up new-style model directories under: ${MD}"
echo ""

# DoFNet — single .pth file → symlink
echo "[dofnet]"
if [ -f "$DOFNET_CKPT" ]; then
    ln -sf "$DOFNET_CKPT" "${MD}/dofnet/pytorch_model.bin"
    ok "linked pytorch_model.bin -> ${DOFNET_CKPT}"
else
    warn "DoFNet ckpt not found at ${DOFNET_CKPT}"
    warn "creating empty marker; edit dofnet/config.json with the correct ckpt_path"
    : > "${MD}/dofnet/pytorch_model.bin"
fi
echo ""

# SAFE — single .pth file → symlink
echo "[safe]"
if [ -f "$SAFE_CKPT" ]; then
    ln -sf "$SAFE_CKPT" "${MD}/safe/pytorch_model.bin"
    ok "linked pytorch_model.bin -> ${SAFE_CKPT}"
else
    warn "SAFE ckpt not found at ${SAFE_CKPT}"
    warn "creating empty marker; edit safe/config.json with the correct ckpt_path"
    : > "${MD}/safe/pytorch_model.bin"
fi
echo ""

# SIDA — sharded weights inside a directory → empty marker
# (real weights are loaded by the adapter from config.sida_path)
echo "[sida]"
if [ -d "$SIDA_DIR" ]; then
    : > "${MD}/sida/pytorch_model.bin"
    ok "created marker; SIDA weights at ${SIDA_DIR} (configured in sida/config.json)"
else
    warn "SIDA dir not found at ${SIDA_DIR}"
    warn "creating empty marker anyway; edit sida/config.json -> sida_path"
    : > "${MD}/sida/pytorch_model.bin"
fi
echo ""

# AeroBlade — three VAE dirs → empty marker
# (real weights are loaded from aeroblade/config.json -> autoencoders[])
echo "[aeroblade]"
missing=0
for d in "$AERO_SD1" "$AERO_SD2" "$AERO_KD21"; do
    if [ ! -d "$d" ]; then
        warn "VAE dir not found: $d"
        missing=$((missing+1))
    fi
done
: > "${MD}/aeroblade/pytorch_model.bin"
if [ "$missing" -eq 0 ]; then
    ok "created marker; 3 VAEs configured in aeroblade/config.json"
else
    warn "${missing} VAE dir(s) missing — fix aeroblade/config.json before launch"
fi
echo ""

echo "Done."
echo ""
echo "Next: run './start.sh' (no args) to verify all 11 models are discovered,"
echo "      then './start.sh start' to launch the service."
