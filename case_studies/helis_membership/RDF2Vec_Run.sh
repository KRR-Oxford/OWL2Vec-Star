#!/bin/bash

echo "rdf2vec, wl, 2"
python -u Baselines.py --embedding_type rdf2vec --walker wl --walk_depth 2

echo "rdf2vec, random, 2"
python -u Baselines.py --embedding_type rdf2vec --walker random --walk_depth 2
