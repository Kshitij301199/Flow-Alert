#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=xlstm            # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-9%5                # job array id

#SBATCH --mem-per-cpu=64G              # Memory Request (per CPU; can use on GLIC)
#SBATCH --gres=gpu:A30:1             # load GPU A100 could be replace by A40/A30, 509-510 nodes has 4_A100_80G
#SBATCH --reservation=GPU              # reserve the GPU
#SBATCH --mail-type=all
#SBATCH --mail-user=kshitkar@gfz-potsdam.de
#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs


# GFZ Configuration with GPUs
module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
source /home/kshitkar/miniforge3/bin/activate
conda activate xlstm_env


parameters1=("ILL18" "ILL12" "ILL13") # input station
parameters2=("xLSTM") # input model
parameters3=("A" "B" "C") # input features

output_dir="/storage/vast-gfz-hpc-01/home/kshitkar/Flow-Alert/trained_model/xlstmmodel"


# Calculate the indices for the current combination
parameters1_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ( ${#parameters2[@]} * ${#parameters3[@]} ) % ${#parameters1[@]} + 1 ))
parameters2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ${#parameters3[@]} % ${#parameters2[@]} + 1 ))
parameters3_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameters3[@]} + 1 ))

# Get the current parameter values
current_parameters1=${parameters1[$parameters1_idx - 1]}
current_parameters2=${parameters2[$parameters2_idx - 1]}
current_parameters3=${parameters3[$parameters3_idx - 1]}

echo "Running job array ID: $SLURM_ARRAY_TASK_ID"
echo "Current parameters: input_station=$current_parameters1, model_type=$current_parameters2, feature_type=$current_parameters3"

# Run your Python script using srun with the parameters
srun --gres=gpu:A30:1 python ../../../functions/xlstm_main.py \
     --output_dir "$output_dir" \
     --input_station "$current_parameters1" \
     --model_type "$current_parameters2" \
     --feature_type "$current_parameters3" \
     --input_component "EHZ" \
     --seq_length 32 \
     --batch_size 16

