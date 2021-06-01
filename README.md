## Dependences 
Our codes in this package have been tested with
  1. Python 3.7
  2. RDFLib 4.2.2
  3. gensim 3.8.0 (Note that gensim 4.x brings important changes, see documention [here](https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4))
  4. scikit-learn 0.21.2
  5. nltk 3.5
  6. OWLready2 0.25
  
 Acknowledgement: 
 Codes under rdf2vec/, which mainly implement walking strategies over RDF graph, 
 come from [pyRDF2Vec](https://github.com/IBCNServices/pyRDF2Vec) (version 0.0.3, accessed in 03/2020), with some updates made.

## Standalone Application
The standalone application (v0.1.0) can be run by the two main programs.

1. **OWL2Vec\_Standalone.py**

    This program will embed one ontology. It can be configured by the configuration file default.cfg. See the examples and comments in default.cfg for the usage.

    Running example: ```python --config_file default.cfg```

    Note: Different from the experimental codes, the standalone program has implemented all OWL ontology relevant procedures in python with Owlready, but it also allows the user to use pre-calculated annotations/axioms/entities/projection for generating the corpus. 

2. **OWL2Vec\_Standalone_Multi.py**

    This program will embed multiple ontologies into one language model, where the documents from multiple ontologies will be merged. One use case example is embedding all the conference relevant ontologies of the OAEI conference track by once.

    Running example: ```python --config_file default_multi.cfg```

    Note: Different from OWL2Vec\_Standalone.py, this program for multiple ontologies does NOT allow the pre-calculated or external annotations/axioms/entities/projection.

## Publications

### Main Reference

- Jiaoyan Chen, Pan Hu, Ernesto Jimenez-Ruiz, Ole Magnus Holter, Denvar Antonyrajah, and Ian Horrocks. [****OWL2Vec\*: Embedding of OWL ontologies****](https://arxiv.org/abs/2009.14654). Machine Learning, Springer, 2021. (accepted). [Codes](https://github.com/KRR-Oxford/OWL2Vec-Star/releases/tag/OWL2Vec-Star-ML-2021-Journal) for the computed results. 

### Applications with OWL2Vec\*
- Jiaoyan Chen, Ernesto Jimenez-Ruiz, Ian Horrocks, Denvar Antonyrajah, Ali Hadian, Jaehun Lee. **Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision**. European Semantic Web Conference, ESWC 2021. ([PDF](https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf)) ([LogMap Matcher work](https://github.com/ernestojimenezruiz/logmap-matcher/))
- Ashley Ritchie, Jiaoyan Chen, Leyla Jael Castro, Dietrich Rebholz-Schuhmann, Ernesto Jiménez-Ruiz. **Ontology Clustering with OWL2Vec\***. DeepOntonNLP ESWC Workshop 2021. ([PDF](https://openaccess.city.ac.uk/id/eprint/25933/1/OntologyClusteringOWL2Vec_DeepOntoNLP2021.pdf)) 

### Preliminary Publications
- Ole Magnus Holter, Erik Bryhn Myklebust, Jiaoyan Chen and Ernesto Jimenez-Ruiz. **Embedding OWL ontologies with OWL2Vec**. International Semantic Web Conference. Poster & Demos. 2019. ([PDF](https://www.cs.ox.ac.uk/isg/TR/OWL2vec_iswc2019_poster.pdf))
- Ole Magnus Holter. **Semantic Embeddings for OWL 2 Ontologies**. MSc thesis, University of Oslo. 2019. ([PDF](https://www.duo.uio.no/bitstream/handle/10852/69078/thesis_ole_magnus_holter.pdf))([GitLab](https://gitlab.com/oholter/owl2vec))


## Case Studies 
Data and codes for class membership prediction on the Healthy Lifestyles (HeLis) ontology, 
and class subsumption prediction on the food ontology FoodOn and the Gene Ontology (GO), are under the folder **case\_studies/**.
