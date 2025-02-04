#!/bin/bash

PATIENT_ID="DA001023"
SAMPLE_NAME="width[5.0]-overlap[2.5]-preictal[3600.0]-nrand[1000]"
SAMPLE_PATH="./samples/clustering/$PATIENT_ID/$SAMPLE_NAME.h5"
TF_METHOD='cwt'
FEATURE_REPO="./samples/timefreq/$PATIENT_ID"
FEATURE_PATH="$FEATURE_REPO/$SAMPLE_NAME-timefreq[$TF_METHOD].h5"

[ ! -d "$FEATURE_REPO" ] && mkdir "$FEATURE_REPO"

# python extract_features.py \
# --sample_file $SAMPLE_PATH \
# --feature_file $FEATURE_PATH \
# --timefreq_method $TF_METHOD \

