#!/usr/bin/env python

"""Tests for ontology_search.py and evaluate_search.py.

Uses a tiny word2vec model trained on the pizza ontology inside the test,
so no pre-trained model or ./output directory is needed.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from gensim.models import Word2Vec

import evaluate_search
import ontology_search
from ontology_search import (
    SearchIndex,
    build_indices,
    embed_classes,
    embed_text,
    load_ontologies,
    parse_camel_plus,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIZZA_OWL = os.path.join(REPO_ROOT, 'case_studies', 'pizza', 'pizza.owl')
PIZZA_NS = 'http://www.co-ode.org/ontologies/pizza/pizza.owl#'

ALIGNMENT_RDF = """<?xml version='1.0' encoding='utf-8'?>
<rdf:RDF xmlns='http://knowledgeweb.semanticweb.org/heterogeneity/alignment'
         xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
<Alignment>
  <map><Cell>
    <entity1 rdf:resource='{ns}SpicyTopping'/>
    <entity2 rdf:resource='{ns}SpicyTopping'/>
    <relation>=</relation>
  </Cell></map>
  <map><Cell>
    <entity1 rdf:resource='{ns}OnionTopping'/>
    <entity2 rdf:resource='{ns}RedOnionTopping'/>
    <relation>=</relation>
  </Cell></map>
  <map><Cell>
    <entity1 rdf:resource='{ns}hasTopping'/>
    <entity2 rdf:resource='{ns}hasBase'/>
    <relation>=</relation>
  </Cell></map>
  <map><Cell>
    <entity1 rdf:resource='{ns}Pizza'/>
    <entity2 rdf:resource='{ns}Food'/>
    <relation>&lt;</relation>
  </Cell></map>
</Alignment>
</rdf:RDF>
""".format(
    ns=PIZZA_NS
)


class TestTextHelpers(unittest.TestCase):

    def test_parse_camel_plus(self):
        self.assertEqual(
            parse_camel_plus('SpicyTopping'), ['spicy', 'topping']
        )
        self.assertEqual(parse_camel_plus('PCMember'), ['pc', 'member'])
        self.assertEqual(
            parse_camel_plus('Conference_participant'),
            ['conference', 'participant'],
        )
        self.assertEqual(parse_camel_plus('pizza'), ['pizza'])

    def test_embed_text_averages_known_words(self):
        model = Word2Vec(
            [['cheese', 'pizza'], ['onion', 'topping']],
            vector_size=8,
            min_count=1,
            seed=1,
        )
        expected = (model.wv['cheese'] + model.wv['pizza']) / 2
        np.testing.assert_allclose(
            embed_text('cheese pizza unknownword', model), expected, rtol=1e-6
        )
        np.testing.assert_allclose(
            embed_text(['cheese', 'pizza'], model), expected, rtol=1e-6
        )

    def test_embed_text_raises_when_no_word_known(self):
        model = Word2Vec([['cheese']], vector_size=8, min_count=1)
        with self.assertRaises(ValueError):
            embed_text('nothing known here', model)


class TestSearchIndex(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.vectors = {
            name: rng.normal(size=16).astype(np.float32) for name in 'abcd'
        }
        # class 'a' has two rows, like a class with IRI + name embeddings
        embeddings = [(n, v) for n, v in self.vectors.items()] + [
            ('a', -self.vectors['b'])
        ]
        self.index = SearchIndex(embeddings, 16)

    def test_exact_vector_ranks_first_with_cosine_one(self):
        iri, score = self.index.search(self.vectors['c'], k=1)[0]
        self.assertEqual(iri, 'c')
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_scores_are_scale_invariant(self):
        scaled = self.index.search(self.vectors['c'] * 100, k=4)
        plain = self.index.search(self.vectors['c'], k=4)
        self.assertEqual([i for i, _ in scaled], [i for i, _ in plain])
        np.testing.assert_allclose(
            [s for _, s in scaled], [s for _, s in plain], atol=1e-5
        )

    def test_results_deduplicated_by_iri(self):
        results = self.index.search(self.vectors['b'], k=None)
        iris = [iri for iri, _ in results]
        self.assertEqual(len(iris), len(set(iris)))
        self.assertEqual(sorted(iris), ['a', 'b', 'c', 'd'])
        self.assertEqual(self.index.n_classes, 4)


class TestPizzaPipeline(unittest.TestCase):
    """End-to-end on pizza.owl with a tiny model: index, search, evaluate."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.onto_dir = os.path.join(cls.tmp, 'ontologies')
        os.makedirs(cls.onto_dir)
        shutil.copy(PIZZA_OWL, cls.onto_dir)
        with open(os.path.join(cls.onto_dir, 'notes.txt'), 'w') as f:
            f.write('not an ontology')

        cls.ontologies = load_ontologies(cls.onto_dir)
        sentences = [
            [c.iri] + parse_camel_plus(c.name)
            for onto in cls.ontologies
            for c in onto.classes()
        ]
        cls.model = Word2Vec(
            sentences * 5, vector_size=16, min_count=1, seed=42, workers=1
        )
        cls.model_path = os.path.join(cls.tmp, 'tiny_word2vec')
        cls.model.save(cls.model_path)

        cls.alignment_dir = os.path.join(cls.tmp, 'alignments')
        os.makedirs(cls.alignment_dir)
        with open(
            os.path.join(cls.alignment_dir, 'pizza-pizza.rdf'), 'w'
        ) as f:
            f.write(ALIGNMENT_RDF)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_load_ontologies_skips_non_ontology_files(self):
        self.assertEqual(len(self.ontologies), 1)
        self.assertEqual(
            self.ontologies[0].base_iri,
            'http://www.co-ode.org/ontologies/pizza/',
        )

    def test_embed_classes_two_embeddings_per_class(self):
        embeds = embed_classes(self.ontologies, self.model, verbose=False)
        entries = embeds['http://www.co-ode.org/ontologies/pizza/']
        n_classes = len(list(self.ontologies[0].classes()))
        self.assertEqual(len(entries), 2 * n_classes)
        self.assertTrue(all(vec.shape == (16,) for _, vec in entries))

    def test_build_indices_and_search_text(self):
        embeds = embed_classes(self.ontologies, self.model, verbose=False)
        indices = build_indices(embeds, self.model.vector_size)
        self.assertEqual(
            set(indices), {'All', 'http://www.co-ode.org/ontologies/pizza/'}
        )
        self.assertEqual(
            indices['All'].index.ntotal,
            indices['http://www.co-ode.org/ontologies/pizza/'].index.ntotal,
        )

        ontology_search.model = self.model
        ontology_search.name2faiss_index = indices
        results = ontology_search.search_text('SpicyTopping', k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], PIZZA_NS + 'SpicyTopping')

    def test_load_reference_pairs_keeps_only_equivalences(self):
        pairs = evaluate_search.load_reference_pairs(self.alignment_dir)
        self.assertEqual(len(pairs), 3)
        self.assertNotIn((PIZZA_NS + 'Pizza', PIZZA_NS + 'Food'), pairs)

    def test_evaluate(self):
        result = evaluate_search.evaluate(
            self.model_path, self.onto_dir, self.alignment_dir
        )
        self.assertEqual(result['reference_pairs'], 3)
        self.assertEqual(result['class_pairs_evaluated'], 2)
        self.assertEqual(
            result['skipped_non_class_pairs'], 1
        )  # hasTopping/hasBase are properties
        name = result['metrics']['name']
        self.assertEqual(name['all']['n'], 4)  # 2 pairs x 2 directions
        self.assertEqual(
            name['same_name']['hits@1'], 1.0
        )  # identical name -> identical vector
        for mode in evaluate_search.QUERY_MODES:
            for metrics in result['metrics'][mode].values():
                self.assertTrue(0.0 < metrics['mrr'] <= 1.0)


class TestEvaluationHelpers(unittest.TestCase):

    def test_normalise_name(self):
        self.assertEqual(
            evaluate_search.normalise_name('http://cmt#ConferenceChair'),
            evaluate_search.normalise_name('http://edas#Conference_Chair'),
        )
        self.assertNotEqual(
            evaluate_search.normalise_name('http://edas#Attendee'),
            evaluate_search.normalise_name(
                'http://conference#Conference_participant'
            ),
        )

    def test_summarise(self):
        m = evaluate_search.summarise([1, 2, 10, 20])
        self.assertEqual(m['n'], 4)
        self.assertEqual(m['hits@1'], 0.25)
        self.assertEqual(m['hits@5'], 0.5)
        self.assertEqual(m['hits@10'], 0.75)
        self.assertAlmostEqual(m['mrr'], (1 + 0.5 + 0.1 + 0.05) / 4)


if __name__ == '__main__':
    unittest.main()
