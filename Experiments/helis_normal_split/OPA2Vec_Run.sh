#!/bin/bash

#echo "opa2vec, no inference, no pretrain, concatenate"
#python -u Baselines.py --embedding_type opa2vec --axiom_file axioms.txt --annotation_file annotations.txt --pretrained none --input_type concatenate

#echo "opa2vec, hermit inference, no pretrain, concatenate"
#python -u Baselines.py --embedding_type opa2vec --axiom_file axioms_hermit.txt --annotation_file annotations.txt --pretrained none --input_type concatenate

#echo "opa2vec, hermit inference, pretrain, concatenate"
#python -u Baselines.py --embedding_type opa2vec --axiom_file axioms_hermit.txt --annotation_file annotations.txt --embedsize 200 --pretrained ~/word2vec/w2v_model/enwiki_model/word2vec_gensim --input_type concatenate

#echo "opa2vec, no inference, no pretrain, minus"
#python -u Baselines.py --embedding_type opa2vec --axiom_file axioms.txt --annotation_file annotations.txt --pretrained none --input_type minus

#echo "opa2vec, hermit inference, no pretrain, minus"
#python -u Baselines.py --embedding_type opa2vec --axiom_file axioms_hermit.txt --annotation_file annotations.txt --pretrained none --input_type minus

#echo "opa2vec, hermit inference, pretrain, minus"
#python -u Baselines.py --embedding_type opa2vec --axiom_file axioms_hermit.txt --annotation_file annotations.txt --embedsize 200 --pretrained ~/word2vec/w2v_model/enwiki_model/word2vec_gensim --input_type minus


echo "opa2vec, no inference, no pretrain, concatenate, english annotations"
python -u Baselines.py --embedding_type opa2vec --axiom_file axioms.txt --annotation_file annotations.txt --pretrained none --input_type concatenate

echo "opa2vec, hermit inference, no pretrain, concatenate, english annotations"
python -u Baselines.py --embedding_type opa2vec --axiom_file axioms_hermit.txt --annotation_file annotations.txt --pretrained none --input_type concatenate

echo "opa2vec, hermit inference, pretrain, concatenate, english annotations"
python -u Baselines.py --embedding_type opa2vec --axiom_file axioms_hermit.txt --annotation_file annotations.txt --embedsize 200 --pretrained ~/word2vec/w2v_model/enwiki_model/word2vec_gensim --input_type concatenate

