# %% [markdown]
# Load OWL2Vec* ontology embeddings into FAISS search indices:
# one index over the classes of all ontologies, plus one index per ontology.
# Run cell by cell in VS Code (Python interactive) with the .venv interpreter,
# or as a script: python ontology_search.py

# %% Imports and helpers
import os
import re
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import faiss
import gensim
import numpy as np
import owlready2
from gensim.models.word2vec import Word2Vec

MODEL_PATH = './output/multi_word2vec'
ONTOLOGY_DIR = './ontologies'
ONTOLOGY_EXTENSIONS = ('.owl', '.rdf', '.ttl', '.xml')


def parse_camel_plus(string: str) -> List[str]:
    """Split a CamelCase / snake_case name into lower-cased words."""
    matches = re.finditer(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', string
    )
    words = [w for m in matches for w in m.group(0).replace('_', ' ').split()]
    return [word.lower() for word in words]


def embed_text(sentence: Union[str, List[str]], model: Word2Vec) -> np.ndarray:
    """Average the embeddings of the words the model knows."""
    if isinstance(sentence, str):
        sentence = sentence.split()
    vectors = [
        model.wv.get_vector(w) for w in sentence if w in model.wv.key_to_index
    ]
    if not vectors:
        raise ValueError(
            f"no words recognized in the given sentence: {sentence}"
        )
    return np.mean(vectors, axis=0).astype(np.float32)


def load_ontologies(
    ontology_dir: str = ONTOLOGY_DIR,
) -> List[owlready2.Ontology]:
    return [
        owlready2.get_ontology(
            os.path.abspath(os.path.join(ontology_dir, f))
        ).load()
        for f in sorted(os.listdir(ontology_dir))
        if f.endswith(ONTOLOGY_EXTENSIONS)
    ]


def embed_classes(
    ontologies: List[owlready2.Ontology], model: Word2Vec, verbose: bool = True
) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """Map ontology IRI -> list of (class IRI, embedding).

    Each class gets two embeddings: its IRI vector and its name vector.
    """
    onto_iri2embeds: Dict[str, List[Tuple[str, np.ndarray]]] = dict()
    for onto in ontologies:
        entries = onto_iri2embeds.setdefault(onto.base_iri, [])
        classes = list(onto.classes())
        for cls in classes:
            if cls.iri in model.wv.key_to_index:
                entries.append((cls.iri, model.wv.get_vector(cls.iri)))
            try:
                entries.append(
                    (cls.iri, embed_text(parse_camel_plus(cls.name), model))
                )
            except ValueError:
                pass
        if verbose:
            print(
                f'{onto.base_iri}: {len(entries)} embeddings '
                f'for {len(classes)} classes'
            )
    return onto_iri2embeds


# %% FAISS index wrapper
class SearchIndex:
    """A FAISS cosine-similarity index plus the class IRI of each row."""

    def __init__(self, embeddings: List[Tuple[str, np.ndarray]], dim: int):
        self.iris = [iri for iri, _ in embeddings]
        self.n_classes = len(set(self.iris))
        vectors = np.stack([vec for _, vec in embeddings]).astype(np.float32)
        faiss.normalize_L2(vectors)  # inner product == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def search(
        self, query: np.ndarray, k: Optional[int] = 5
    ) -> List[Tuple[str, float]]:
        """Top-k classes for a query vector; k=None ranks every class."""
        k = self.n_classes if k is None else k
        q = query.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(q)
        # each class has up to 2 rows, so over-fetch then dedupe by IRI
        scores, ids = self.index.search(q, min(len(self.iris), 2 * k))
        results, seen = [], set()
        for score, i in zip(scores[0], ids[0]):
            if i >= 0 and self.iris[i] not in seen:
                seen.add(self.iris[i])
                results.append((self.iris[i], float(score)))
        return results[:k]


def build_indices(
    onto_iri2embeds: Dict[str, List[Tuple[str, np.ndarray]]], dim: int
) -> Dict[str, SearchIndex]:
    """'All' index over every class, plus one index per ontology IRI."""
    name2faiss_index: Dict[str, SearchIndex] = dict()
    name2faiss_index['All'] = SearchIndex(
        list(chain.from_iterable(onto_iri2embeds.values())), dim
    )
    for ontology_iri, embeddings in onto_iri2embeds.items():
        if embeddings:
            name2faiss_index[ontology_iri] = SearchIndex(embeddings, dim)
    return name2faiss_index


def search_text(
    text: str, index_name: str = 'All', k: int = 5
) -> List[Tuple[str, float]]:
    """Text query against the module-level indices built in the cell below."""
    words = [w for token in text.split() for w in parse_camel_plus(token)]
    return name2faiss_index[index_name].search(embed_text(words, model), k)


# %% Load model and ontologies, build indices
if __name__ == '__main__':
    model = gensim.models.Word2Vec.load(MODEL_PATH)
    ontologies = load_ontologies(ONTOLOGY_DIR)
    onto_iri2embeds = embed_classes(ontologies, model)
    name2faiss_index = build_indices(onto_iri2embeds, model.vector_size)
    print(
        'Indices:',
        {name: idx.index.ntotal for name, idx in name2faiss_index.items()},
    )

# %% Example queries
if __name__ == '__main__':
    for query in ['conference chair', 'paper review', 'accepted paper']:
        print(f'\n{query!r}:')
        for iri, score in search_text(query):
            print(f'  {score:.3f}  {iri}')
