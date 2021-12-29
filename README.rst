============
OWL2Vec-Star
============


.. |pypi|  image:: https://img.shields.io/pypi/v/owl2vec-star.svg
           :target: https://pypi.python.org/pypi/owl2vec-star

.. |docs|  image:: https://readthedocs.org/projects/owl2vec-star/badge/?version=latest
           :target: https://owl2vec-star.readthedocs.io/en/latest/?version=latest
           :alt: Documentation Status

|pypi| |docs|

**OWL2Vec*: Embedding OWL ontologies**


* Free software: Apache-2.0 License
* Documentation: https://owl2vec-star.readthedocs.io.


Features
--------

OWL2Vec* v0.2.0 exposes a CLI with two subcommands after installation, which allows you to perform two main programs.
You can also run the two original python programs without installation (see the requirements in `setup.py <https://github.com/KRR-Oxford/OWL2Vec-Star/blob/master/setup.py>`__).

Installation command::

    $ make install

Standalone
~~~~~~~~~~~~~~~~~~~~~~

This command will embed one ontology. It can be configured by the configuration file default.cfg.
See the examples and comments in default.cfg for the usage.

Running command::

    $ owl2vec_star standalone --config_file default.cfg

Running program::

    $ python OWL2Vec_Standalone.py --config_file default.cfg


Note: Different from the experimental codes, the standalone command has implemented all OWL ontology
relevant procedures in python with Owlready, but it also allows the user to use pre-calculated
annotations/axioms/entities/projection to generate the corpus.

Standalone Multi
~~~~~~~~~~~~~~~~

This command will embed multiple ontologies into one embedding model, where the documents from
multiple ontologies will be merged. One use case example is embedding all the conference relevant
ontologies of the OAEI conference track at once.

Running command::

    $ owl2vec_star standalone-multi --config_file default_multi.cfg

Running program::

    $ python OWL2Vec_Standalone.py --config_file default_multi.cfg

Note: Different from the `standalone` command, this command for multiple ontologies does NOT allow
the pre-calculated or external annotations/axioms/entities/projection.

Publications
------------

Main Reference
~~~~~~~~~~~~~~

* Jiaoyan Chen, Pan Hu, Ernesto Jimenez-Ruiz, Ole Magnus Holter, Denvar Antonyrajah, and Ian Horrocks.
  **OWL2Vec*: Embedding of OWL ontologies**. Machine Learning, Springer, 2021.
  [`PDF <https://arxiv.org/abs/2009.14654>`_]
  [`@Springer <https://rdcu.be/cmIMh>`_] 
  [`Collection <https://link.springer.com/journal/10994/topicalCollection/AC_f13088dda1f43d317c5acbfdf9439a31>`_]
  [`Codes <https://github.com/KRR-Oxford/OWL2Vec-Star/releases/tag/OWL2Vec-Star-ML-2021-Journal>`__
  for the computed results.]


Applications with OWL2Vec*
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Jiaoyan Chen, Ernesto Jimenez-Ruiz, Ian Horrocks, Denvar Antonyrajah, Ali Hadian, Jaehun Lee.
  **Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision**.
  European Semantic Web Conference, ESWC 2021.
  [`PDF <https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf>`__]
  [`LogMap Matcher work <https://github.com/ernestojimenezruiz/logmap-matcher/>`__]
- Ashley Ritchie, Jiaoyan Chen, Leyla Jael Castro, Dietrich Rebholz-Schuhmann, Ernesto Jiménez-Ruiz.
  **Ontology Clustering with OWL2Vec\***.
  DeepOntonNLP ESWC Workshop 2021.
  [`PDF <https://openaccess.city.ac.uk/id/eprint/25933/1/OntologyClusteringOWL2Vec_DeepOntoNLP2021.pdf>`__]

Preliminary Publications
~~~~~~~~~~~~~~~~~~~~~~~~
- Ole Magnus Holter, Erik Bryhn Myklebust, Jiaoyan Chen and Ernesto Jimenez-Ruiz.
  **Embedding OWL ontologies with OWL2Vec**.
  International Semantic Web Conference.
  Poster & Demos. 2019.
  [`PDF <https://www.cs.ox.ac.uk/isg/TR/OWL2vec_iswc2019_poster.pdf>`__]
- Ole Magnus Holter. **Semantic Embeddings for OWL 2 Ontologies**.
  MSc thesis, University of Oslo. 2019.
  [`PDF <https://www.duo.uio.no/bitstream/handle/10852/69078/thesis_ole_magnus_holter.pdf>`__]
  [`GitLab <https://gitlab.com/oholter/owl2vec>`__]


Case Studies
------------
Data and codes for class membership prediction on the Healthy Lifestyles (HeLis) ontology,
and class subsumption prediction on the food ontology FoodOn and the Gene Ontology (GO), are under the
folder `case_studies/`.


Credits
-------
Code under `owl2vec_star/rdf2vec/`, which mainly implement walking strategies over RDF graphs,
is derived from `pyRDF2Vec`_ (version 0.0.3, last access: 03/2020) with revision.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
Many thanks to `Vincenzo Cutrona <https://github.com/vcutrona>`_ for preparing this package.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _`pyRDF2Vec`: https://github.com/IBCNServices/pyRDF2Vec
