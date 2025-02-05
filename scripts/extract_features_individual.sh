#!/bin/bash

read -p "Enter the patient ID: " PATIENT_ID
read -p "Enter the preictal value: " PREICTAL
read -p "Enter the nrand value: " NRAND
SAMPLE_NAME="width[5.0]-overlap[2.5]-preictal[$PREICTAL]-nrand[$NRAND]"
SAMPLE_PATH="./samples/clustering/$PATIENT_ID/$SAMPLE_NAME.h5"
TF_METHOD='cwt'
FEATURE_REPO="./samples/timefreq/$PATIENT_ID"
FEATURE_PATH="$FEATURE_REPO/$SAMPLE_NAME-timefreq[$TF_METHOD].h5"

[ ! -d "$FEATURE_REPO" ] && mkdir "$FEATURE_REPO"

python extract_features.py \
--sample_file $SAMPLE_PATH \
--feature_file $FEATURE_PATH \
--timefreq_method $TF_METHOD
