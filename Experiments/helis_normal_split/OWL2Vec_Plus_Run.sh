#!/bin/bash

echo "OWL2Vec Plus, URI_Doc, URI"
python -u OWL2Vec_Plus.py --URI_Doc yes --Lit_Doc no --Mix_Doc no --Embed_Out_URI yes --Embed_Out_Words no

echo "OWL2Vec Plus, URI_Doc + Lit_Doc, Word"
python -u OWL2Vec_Plus.py --URI_Doc yes --Lit_Doc yes --Mix_Doc no --Embed_Out_URI no --Embed_Out_Words yes

echo "OWL2Vec Plus, URI_Doc + Lit_Doc + Mix_Doc, random, Word"
python -u OWL2Vec_Plus.py --URI_Doc yes --Lit_Doc yes --Mix_Doc yes --Mix_Type random --Embed_Out_URI no --Embed_Out_Words yes

echo "OWL2Vec Plus, URI_Doc + Lit_Doc + Mix_Doc, random, Word + URI"
python -u OWL2Vec_Plus.py --URI_Doc yes --Lit_Doc yes --Mix_Doc yes --Mix_Type random --Embed_Out_URI yes --Embed_Out_Words yes
