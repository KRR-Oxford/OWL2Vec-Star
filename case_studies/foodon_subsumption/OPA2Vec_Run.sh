#!/bin/bash

echo "opa2vec, no inference, no pretrain"
python -u Baselines.py --embedding_type opa2vec --axiom_file axioms.txt --annotation_file annotations.txt --pretrained none

echo "opa2vec, hermit inference, no pretrain"
python -u Baselines.py --embedding_type opa2vec --axiom_file axioms_hermit.txt --annotation_file annotations.txt --pretrained none

echo "opa2vec, hermit inference, pretrain"
python -u Baselines.py --embedding_type opa2vec --axiom_file axioms_hermit.txt --annotation_file annotations.txt --embedsize 200 --pretrained ~/word2vec/w2v_model/enwiki_model/word2vec_gensim


