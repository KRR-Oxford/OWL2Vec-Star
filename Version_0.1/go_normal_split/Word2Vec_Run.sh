#!/bin/bash

echo "Word2Vec, concatenate"
python -u OWL2Vec_Plus.py --URI_Doc no --Lit_Doc no --Mix_Doc no --Embed_Out_URI no --Embed_Out_Words yes --embedsize 200 --pretrained ~/word2vec/w2v_model/enwiki_model/word2vec_gensim


