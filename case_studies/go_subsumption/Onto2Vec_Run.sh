#!/bin/bash

#echo "onto2vec, no inference, no pretrain"
#python -u Baselines.py --embedding_type onto2vec --axiom_file axioms.txt --pretrained none

echo "onto2vec, hermit inference, no pretrain"
python -u Baselines.py --embedding_type onto2vec --axiom_file axioms_hermit.txt --pretrained none



