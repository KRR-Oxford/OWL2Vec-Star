#!/bin/bash

echo "onto2vec, no inference, no pretrain, concatenate"
python -u Baselines.py --embedding_type onto2vec --axiom_file axioms.txt --pretrained none --input_type concatenate

echo "onto2vec, hermit inference, no pretrain, concatenate"
python -u Baselines.py --embedding_type onto2vec --axiom_file axioms_hermit.txt --pretrained none --input_type concatenate

echo "onto2vec, no inference, no pretrain, minus"
python -u Baselines.py --embedding_type onto2vec --axiom_file axioms.txt --pretrained none --input_type minus

echo "onto2vec, hermit inference, no pretrain, minus"
python -u Baselines.py --embedding_type onto2vec --axiom_file axioms_hermit.txt --pretrained none --input_type minus



