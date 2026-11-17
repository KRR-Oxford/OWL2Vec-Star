=================================
Ontology Search and Evaluation
=================================

This guide covers three additions built on top of OWL2Vec* Standalone Multi:

1. **Fine-tuning** a pre-trained word2vec model on a catalog of ontologies.
2. **Semantic search** over ontology classes with FAISS
   (``ontology_search.py``).
3. **Evaluation** of search quality against the OAEI Conference reference
   alignments (``evaluate_search.py``).

.. contents::
   :local:
   :depth: 1


Setup
-----

The code is tested with Python 3.9. The pinned ``numpy``/``scipy`` versions in
``requirements.txt`` do not install on Python 3.12.

::

    $ pyenv install 3.9.19
    $ ~/.pyenv/versions/3.9.19/bin/python -m venv .venv
    $ source .venv/bin/activate
    $ pip install -r requirements.txt
    $ python -c "import nltk; nltk.download('punkt')"

``requirements.txt`` pins ``faiss-cpu==1.8.0``. Do not install a newer
``faiss-cpu`` on its own: releases from 1.9 upwards pull in ``numpy`` 2.x,
which breaks the pinned ``gensim``, ``scipy`` and ``scikit-learn``.

Pre-trained word2vec model
~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the English Wikipedia word2vec model (200 dimensions, ~3.5 GB
unpacked) from https://tinyurl.com/word2vec-model and extract it so that
``word2vec_gensim`` and its ``.npy`` files sit in the same directory, e.g.::

    ~/w2v_model/enwiki_model/
        word2vec_gensim
        word2vec_gensim.wv.vectors.npy
        word2vec_gensim.trainables.syn1neg.npy

Then, in ``default_multi.cfg``, uncomment ``pre_train_model`` and point it at
the ``word2vec_gensim`` file, and set the fine-tuning epochs used for the
results below::

    pre_train_model = ~/w2v_model/enwiki_model/word2vec_gensim
    epoch = 200

If ``pre_train_model`` does not exist, the script prints a warning and trains
from scratch instead of fine-tuning. ``embed_size`` is ignored when
fine-tuning; the pre-trained model's size (200) is used.


Data
----

Download the `OAEI Conference track
<https://oaei.ontologymatching.org/2023/conference/>`__ ontologies and
reference alignments::

    $ python download_oaei_conference.py

``ontologies/``
    Input catalog: the 16 OAEI Conference ontologies (cmt, edas, ekaw, ...).
    Every ``.owl``/``.rdf``/``.ttl``/``.xml`` file in this directory is
    embedded, so you can add your own, e.g.
    ``cp case_studies/pizza/pizza.owl ontologies/``.

``reference_alignments/conference/``
    The 21 OAEI Conference reference alignment files: 305 equivalence pairs
    among cmt, conference, confOf, edas, ekaw, iasted and sigkdd.
    Used only for evaluation.

``output/``
    Trained models and evaluation history. A fine-tuned model is ~1.4 GB,
    almost all of it the pre-trained English vocabulary.

All three directories are ignored by git.


1. Train
--------

::

    $ python OWL2Vec_Standalone_Multi.py --URI_Doc --Lit_Doc --Mix_Doc

``default_multi.cfg`` reads ontologies from ``./ontologies/`` and saves the
model to ``./output/multi_word2vec``; ``--ontology_dir`` and
``--embedding_dir`` override them.

On an Apple Silicon laptop, training on the 16 Conference ontologies takes
about 2 minutes (half building the corpus, half fine-tuning).

The script always retrains on every ontology in the directory; there is no
incremental update. Vectors from different training runs are not comparable,
so after any retrain **all** search indices must be rebuilt.


2. Search
---------

``ontology_search.py`` embeds every class twice and loads the vectors into
FAISS:

* **IRI vector**: the class's own OWL2Vec* embedding.
* **Name vector**: the average word vector of the class name, split from
  CamelCase/snake_case (``ConferenceChair`` -> ``conference chair``).

It builds an ``All`` index over every class, plus one index per ontology,
keyed by the ontology's base IRI. Scores are cosine similarity (vectors are
L2-normalised; 1.0 means identical direction). Each class appears at most
once in a result list.

Run the example queries::

    $ python ontology_search.py

Or use it from Python / a notebook (the ``# %%`` cells run in VS Code's
interactive window)::

    import gensim
    from ontology_search import (load_ontologies, embed_classes,
                                 build_indices, embed_text, parse_camel_plus)

    model = gensim.models.Word2Vec.load('./output/multi_word2vec')
    ontologies = load_ontologies('./ontologies')
    indices = build_indices(embed_classes(ontologies, model),
                            model.vector_size)

    query = embed_text(parse_camel_plus('ConferenceChair'), model)
    indices['All'].search(query, k=5)          # across every ontology
    indices['http://edas#'].search(query, k=5)  # one ontology only

Example results (16 Conference ontologies, all queries against ``All``):

==================  ==========================================================
Query               Top results
==================  ==========================================================
conference chair    ``edas#ConferenceChair`` 1.00, ``cmt#ConferenceChair``
                    1.00, ``crs_dr#chair`` 0.87, ``myreview#Chair`` 0.87
paper review        ``openconf#Paper_Review`` 1.00, ``ekaw#Review`` 0.83,
                    ``edas#Review`` 0.83
accepted paper      ``ekaw#Accepted_Paper``, ``edas#AcceptedPaper``,
                    ``confious#Accepted_Paper``, ``pcs#Accepted_paper`` (all
                    1.00)
==================  ==========================================================

``IndexFlatIP`` performs exact search and is fast up to about a million
vectors. For larger catalogs, switch ``SearchIndex`` to an approximate index
such as ``faiss.IndexHNSWFlat``.


3. Evaluate
-----------

``evaluate_search.py`` measures whether search finds the class that the OAEI
reference says is equivalent. For each class pair ``A = B`` it queries with
``A`` and records where ``B`` ranks in its own ontology's index, then repeats
in the other direction.

::

    $ python evaluate_search.py --label "what changed in this run"

It reports Hits@1, Hits@5, Hits@10 and MRR, broken down by:

* **query vector**: ``iri``, ``name``, or ``both`` (mean of the two);
* **pair type**: ``same_name`` (e.g. ``cmt#Person = edas#Person``, which any
  string match finds) and ``different_name`` (e.g.
  ``conference#Conference_participant = edas#Attendee``, the real test).

Pairs involving properties rather than classes are skipped (46 of 305). Each
run is appended to ``output/eval_history.jsonl`` and printed with the MRR
change against the previous run.

Baseline
~~~~~~~~

16 Conference ontologies, pre-trained enwiki model, ``epoch = 200``, cosine
scoring; 259 class pairs x 2 directions:

=======  ===============  ======  ======  =======  =====
Query    Pairs            Hits@1  Hits@5  Hits@10  MRR
=======  ===============  ======  ======  =======  =====
iri      all              0.139   0.299   0.448    0.237
name     all              0.759   0.867   0.921    0.810
name     different_name   0.479   0.713   0.829    0.590
both     different_name   0.412   0.625   0.742    0.518
=======  ===============  ======  ======  =======  =====

Random walks make each training run slightly different; expect these numbers
to vary by a few points between runs.

The fine-tuned IRI vectors find the matching class first only 14% of the
time, and adding them to the name vector lowers the score. Training places
each class near the rest of its *own* ontology rather than near its equivalent
elsewhere. Search quality across ontologies therefore comes almost entirely
from the class-name vectors.

Suggested experiments, each a retrain plus a ~4 s evaluation:

* index only the name vectors;
* change ``walk_depth`` or disable ``Mix_Doc``;
* compare against the pre-trained model with no fine-tuning.

Suggested update workflow: add ontologies, retrain, rebuild indices, run
``evaluate_search.py``, and promote the new model only if
``different_name`` MRR holds up.


Tests
-----

::

    $ flake8 ontology_search.py evaluate_search.py tests/test_ontology_search.py
    $ pytest tests/test_ontology_search.py

The tests train a tiny word2vec model on ``case_studies/pizza/pizza.owl``
inside the test, so they need neither the pre-trained model nor ``output/``,
and they finish in about a second.
