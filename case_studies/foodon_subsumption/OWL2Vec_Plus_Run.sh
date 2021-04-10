#!/bin/bash

echo "OWL2Vec Plus, URI_Doc, URI, wl, depth=4, URI Vector"
python -u OWL2Vec_Plus.py --walker wl --walk_depth 4 --URI_Doc yes --Lit_Doc no --Embed_Out_URI yes --Embed_Out_Words no

echo "OWL2Vec Plus, URI_Doc + Lit_Doc, URI, wl, depth=4, Word Vector"
python -u OWL2Vec_Plus.py --walker wl --walk_depth 4 --URI_Doc yes --Lit_Doc yes --Embed_Out_URI no --Embed_Out_Words yes


