#!/bin/sh
# ===========================================================================
#  install-linux.sh -- NoLlama setup entry point on Linux
#
#  There is no separate Linux installer: install.ps1 is cross-platform and
#  runs under PowerShell 7 (pwsh). This script checks that pwsh and Python
#  exist, tells you the exact command to install them if not, then hands off.
#
#  Confirmed working: Ubuntu on Core Ultra 7 258V (NPU + GPU), issue #6.
# ===========================================================================
set -eu
cd "$(dirname "$0")"

echo ""
echo " === NoLlama install (Linux) ==="
echo ""

# --- 1. PowerShell 7 --------------------------------------------------------
if ! command -v pwsh >/dev/null 2>&1; then
    echo " PowerShell 7 (pwsh) is not installed. NoLlama's installer runs on it."
    echo ""
    echo " Install it with ONE of these, then re-run ./install-linux.sh:"
    echo ""
    if command -v snap >/dev/null 2>&1; then
        echo "   sudo snap install powershell --classic"
        echo ""
    fi
    if command -v apt >/dev/null 2>&1; then
        echo "   # Ubuntu/Debian via Microsoft's repo:"
        echo "   wget -q https://packages.microsoft.com/config/ubuntu/\$(lsb_release -rs)/packages-microsoft-prod.deb"
        echo "   sudo dpkg -i packages-microsoft-prod.deb && sudo apt update && sudo apt install -y powershell"
        echo ""
    fi
    if command -v dnf >/dev/null 2>&1; then
        echo "   # Fedora/RHEL via Microsoft's repo:"
        echo "   sudo dnf install -y https://packages.microsoft.com/config/rhel/9/packages-microsoft-prod.rpm"
        echo "   sudo dnf install -y powershell"
        echo ""
    fi
    echo "   All options: https://learn.microsoft.com/powershell/scripting/install/installing-powershell-on-linux"
    echo ""
    exit 1
fi
echo " [+] PowerShell 7 found."

# --- 2. Python 3.10+ ---------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1 || ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    echo " Python 3.10+ not found. Install it with your package manager, e.g.:"
    echo ""
    echo "   sudo apt install python3 python3-venv     # Ubuntu/Debian"
    echo "   sudo dnf install python3                  # Fedora/RHEL"
    echo ""
    exit 1
fi
echo " [+] Python 3.10+ found."

# --- 3. Heads-up: Intel drivers ------------------------------------------------
# Without the Intel userspace drivers, only the CPU is detected. That works,
# but it is not why you are here. The NPU stack on Linux is younger than on
# Windows -- see https://github.com/intel/linux-npu-driver (NPU) and your
# distro's intel-opencl/compute-runtime packages (GPU).
echo " [i] If install.ps1 detects only CPU: install Intel's GPU compute runtime"
echo "     and/or NPU driver first (see comments in this script)."

# --- 4. Hand off ---------------------------------------------------------------
echo ""
echo " Handing off to install.ps1 (device detection + model menu)..."
echo ""
exec pwsh -NoLogo -File ./install.ps1 "$@"
