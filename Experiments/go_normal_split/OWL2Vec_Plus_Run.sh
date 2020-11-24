#!/bin/bash

#echo "OWL2Vec Plus, URI_Doc + Lit_Doc, Word"
#python -u OWL2Vec_Plus.py --URI_Doc yes --Lit_Doc yes --Mix_Doc no --Embed_Out_URI no --Embed_Out_Words yes

echo "OWL2Vec (original), URI_Doc, URI, concatenate, wl 2, projection"
python -u OWL2Vec_Plus.py --URI_Doc yes --Lit_Doc no --Mix_Doc no --Embed_Out_URI yes --Embed_Out_Words no  --walker wl --walk_depth 2 --onto_file go.train.projection.ttl

