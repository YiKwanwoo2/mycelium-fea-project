#!/bin/bash
#SBATCH --job-name=mycelium_fea
#SBATCH --account=ners570f25_class          
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================
# 0. Setup Environment
# ============================

# Load Python (system module or virtualenv)
module purge
module load python/3.11.5 2>/dev/null || module load python/3.10.13 || true

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

mkdir -p results logs

# ============================
# 1. Run Mycelium Growth Model
# ============================

echo "=== Running mycelium growth simulation ==="
python src/mycelium_sim.py

# Find most recent simulation directory
RESULT_DIR=$(ls -td results/sim_* | head -n 1)

if [ ! -d "$RESULT_DIR" ]; then
    echo "No results directory found! Exiting..."
    exit 1
fi
echo "Found results directory: $RESULT_DIR"

# ============================
# 2. Run Finite Element Solver
# ============================

echo "=== Running FEA solver ==="
python src/fea_solver.py "$RESULT_DIR"

echo "Simulation and FEA complete!"
