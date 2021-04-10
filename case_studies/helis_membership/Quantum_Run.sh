#!/bin/bash

echo "Quantum, concatenate, 27000 iterations"
python -u Baselines.py --embedding_type quantum --q_embedding_class_file qembeddings-27000iters/helisclassembeddings.txt --q_embedding_individual_file qembeddings-27000iters/helisindividualembeddings.txt --input_type concatenate

echo "Quantum, minus, 27000 iterations"
python -u Baselines.py --embedding_type quantum --q_embedding_class_file qembeddings-27000iters/helisclassembeddings.txt --q_embedding_individual_file qembeddings-27000iters/helisindividualembeddings.txt --input_type minus


