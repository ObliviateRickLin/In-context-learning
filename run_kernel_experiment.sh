#!/bin/bash
#$ -cwd
#$ -j y
#$ -o kernel_experiment_output.$JOB_ID
#$ -l gpu,A6000,cuda=1,h_data=16G,h_rt=24:00:00

# Load modules
module load python/3.9.6
module load cuda/11.7
module load anaconda3/2023.03

# Activate virtual environment
source icl_venv/bin/activate

# Download model checkpoints if needed
if [ ! -d "models" ]; then
    wget https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip
    unzip models.zip
fi

# Create output directories
mkdir -p models/kernel_regression
mkdir -p models/kernel_regression_rbf

# Set CUDA config
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Run standard kernel regression experiment
cd src
echo "Starting standard kernel regression experiment..."
python train.py --config conf/kernel_regression.yaml --out_dir ../models/kernel_regression

# Run RBF kernel regression experiment
echo "Starting RBF kernel regression experiment..."
python train.py --config conf/kernel_regression_rbf.yaml --out_dir ../models/kernel_regression_rbf

echo "Experiments completed!"

# For evaluation, you can use:
# python -c "import eval; eval.main()" --model_path ../models/kernel_regression --task kernel_regression
# python -c "import eval; eval.main()" --model_path ../models/kernel_regression_rbf --task kernel_regression 