#!/bin/bash
#SBATCH --job-name=mycelium        # Job name
#SBATCH --account=ners570f25_class
#SBATCH --nodes=1                  # Use 1 node
#SBATCH --ntasks=1                 # 1 task (single process)
#SBATCH --cpus-per-task=4          # Use 4 cores
#SBATCH --mem=8G                   # Memory per node
#SBATCH --time=01:00:00            # Max runtime (HH:MM:SS) 
#SBATCH --output=logs/mycelium_%j.out     # Standard output log (%j = job ID)
#SBATCH --error=logs/mycelium_%j.err      # Error log

# ---- SETUP ENVIRONMENT ----
module load python/3.13.2             # Load Python module
source ~/ners570_project/mycelium-fea-project/venv/bin/activate  # Activate venv

# ---- NAVIGATE TO PROJECT ----
cd ~/ners570_project/mycelium-fea-project

# ---- RUN SCRIPT ----
python src/mycelium_sim.py

