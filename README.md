Our codes in this package are tested with
1. Python 3.7
2. RDFLib 4.2.2
3. gensim 3.8.0
4. scikit-learn 0.21.2
5. nltk 3.5
6. OWLready 0.25

The folder **Experiments/** includes data and codes used in our paper ***"OWL2Vec\*: Embedding OWL Ontologies"***. See https://arxiv.org/abs/2009.14654. 


The standalone application is ready in the folder **Standalone\_0.1/** by the main program OWL2Vec\_Standalone.py.
It can be configured by the configuration file default.cfg. Running example: python --config\_file default.cfg


The standalone can support OWL2Vec\*, as well as its variants such as OWL2Vec, Onto2Vec and OPA2Vec. See comments in OWL2Vec\_Standalone.py and default.cfg. Different from the experimental codes, the standalone program has implemented all OWL ontology relevant procedures in python with Owlready 0.25, but it also allows the user to use pre-calculated annotations/axioms/entities/projection for generating the corpus. More tests and a more concrete document is on the way.

