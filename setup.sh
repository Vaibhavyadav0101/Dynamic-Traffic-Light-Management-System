#!/bin/bash
# ============================================================
# SUMO Traffic RL Project - Setup Script
# Run this once from your project root folder
# ============================================================

echo "========================================="
echo " SUMO Traffic RL Project - Setup"
echo "========================================="

# ---------- Step 1: Install SUMO ----------
echo ""
echo "[1/4] Installing SUMO..."
sudo apt-get update -qq
sudo apt-get install -y sumo sumo-tools sumo-doc

# Set SUMO_HOME environment variable permanently
SUMO_PATH=$(which sumo)
SUMO_DIR=$(dirname $(dirname $SUMO_PATH))
export SUMO_HOME="$SUMO_DIR"

# Add to ~/.bashrc so it persists across terminal sessions
if ! grep -q "SUMO_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# SUMO Traffic Simulator" >> ~/.bashrc
    echo "export SUMO_HOME=\"$SUMO_DIR\"" >> ~/.bashrc
    echo "export PATH=\"\$SUMO_HOME/bin:\$PATH\"" >> ~/.bashrc
fi

echo "    SUMO installed at: $SUMO_DIR"
echo "    SUMO_HOME set to: $SUMO_HOME"

# ---------- Step 2: Install Python packages ----------
echo ""
echo "[2/4] Installing Python packages..."
pip install torch --break-system-packages -q
pip install numpy matplotlib pandas pyserial --break-system-packages -q

# traci comes bundled with SUMO tools - add to PYTHONPATH
SUMO_TOOLS="$SUMO_HOME/tools"
if ! grep -q "SUMO_HOME/tools" ~/.bashrc; then
    echo "export PYTHONPATH=\"\$SUMO_HOME/tools:\$PYTHONPATH\"" >> ~/.bashrc
fi
export PYTHONPATH="$SUMO_TOOLS:$PYTHONPATH"
echo "    traci path: $SUMO_TOOLS"

# ---------- Step 3: Create project folder structure ----------
echo ""
echo "[3/4] Creating folder structure..."
mkdir -p maps maps_images models plots
echo "    Created: maps/, maps_images/, models/, plots/"

# ---------- Step 4: Verify ----------
echo ""
echo "[4/4] Verifying installation..."
sumo --version 2>&1 | head -1
python3 -c "import traci; print('    traci: OK')" 2>/dev/null || echo "    traci: needs SUMO_HOME reload (see below)"
python3 -c "import torch; print(f'    torch: OK (v{torch.__version__})')"
python3 -c "import numpy; print(f'    numpy: OK (v{numpy.__version__})')"
python3 -c "import matplotlib; print(f'    matplotlib: OK (v{matplotlib.__version__})')"
python3 -c "import pandas; print(f'    pandas: OK (v{pandas.__version__})')"

echo ""
echo "========================================="
echo " Setup complete!"
echo "========================================="
echo ""
echo " IMPORTANT: Run this command to reload your environment:"
echo "   source ~/.bashrc"
echo ""
echo " Then run your scripts like this:"
echo ""
echo "   # Train the RL model:"
echo "   python3 train.py --train -e 50 -s 500"
echo ""
echo "   # Test the trained model:"
echo "   python3 train.py -m model"
echo ""
echo "   # Run fixed-time baseline:"
echo "   python3 train_fixed.py -s 1000 -d 30"
echo ""
echo "   # Generate plots:"
echo "   python3 plots_script.py"
echo "========================================="
