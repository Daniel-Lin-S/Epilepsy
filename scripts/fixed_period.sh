#!/bin/bash

echo "Please enter the model name (rf, svm, logreg):"
read model_name

sample_lengths=(5.0 10.0 20.0)
preictal_times=(10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0)

# Loop through all combinations
for sample_length in "${sample_lengths[@]}"; do
    for preictal_time in "${preictal_times[@]}"; do
        python run_classifier.py \
        --sample_length $sample_length \
        --preictal_time $preictal_time \
        --select_channels \
        --model_name $model_name \
        --n_features 100
    done
done