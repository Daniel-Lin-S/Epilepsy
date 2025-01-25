#!/bin/bash

# Define the possible values for sample_length and preictal_time
sample_lengths=(5.0 10.0 20.0)
preictal_times=(10.0 20.0 30.0 40.0 60.0)

# Loop through all combinations
for sample_length in "${sample_lengths[@]}"; do
    for preictal_time in "${preictal_times[@]}"; do
        python run_classifier.py \
        --sample_length $sample_length \
        --preictal_time $preictal_time \
        --select_channels \
        --model_name rf
    done
done