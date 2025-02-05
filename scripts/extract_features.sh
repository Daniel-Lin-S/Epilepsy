#!/bin/bash

read -p "enter the name of h5 sample file (no postfix)" SAMPLE_NAME
SAMPLE_PATH="./samples/clustering/$SAMPLE_NAME.h5"
TF_METHOD='cwt'
FEATURE_REPO="./samples/timefreq"
FEATURE_PATH="$FEATURE_REPO/$SAMPLE_NAME-timefreq[$TF_METHOD].h5"

[ ! -d "$FEATURE_REPO" ] && mkdir "$FEATURE_REPO"

python extract_features.py \
--sample_file $SAMPLE_PATH \
--feature_file $FEATURE_PATH \
--timefreq_method $TF_METHOD
