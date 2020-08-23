#!/bin/bash

echo "EL, concatenate"
python -u Baselines.py --embedding_type el --el_embedding_file el_cls_embeddings.pkl --input_type concatenate

