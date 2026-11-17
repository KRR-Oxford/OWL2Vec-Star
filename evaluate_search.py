"""Evaluate ontology search quality against OAEI reference alignments.

For every reference match (A = B) between two ontologies, query with class A
and check where class B ranks in the target ontology's index (and vice versa).

Metrics: Hits@1, Hits@5, Hits@10 and MRR (mean reciprocal rank), reported for
  - all pairs
  - "same name" pairs (e.g. cmt#Person = edas#Person; trivial for strings)
  - "different name" pairs
    (e.g. conference#Conference_participant = edas#Attendee)
and for three ways of building the query vector:
  - iri:  the class's own OWL2Vec* IRI embedding
  - name: average word embedding of the class name
  - both: mean of the two

Each run is appended to output/eval_history.jsonl and compared with the
previous run.

Usage:
  python evaluate_search.py
  python evaluate_search.py --model ./output/multi_word2vec --label "epoch200"
"""

import argparse
import glob
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import gensim
import numpy as np

from ontology_search import (
    MODEL_PATH,
    ONTOLOGY_DIR,
    build_indices,
    embed_classes,
    embed_text,
    load_ontologies,
    parse_camel_plus,
)

ALIGN_NS = '{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}'
RDF_NS = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
HITS_AT = (1, 5, 10)
QUERY_MODES = ('iri', 'name', 'both')


def load_reference_pairs(alignment_dir: str) -> List[Tuple[str, str]]:
    """Read equivalence (=) correspondences from OAEI alignment RDF files."""
    pairs = []
    for path in sorted(glob.glob(os.path.join(alignment_dir, '*.rdf'))):
        for cell in ET.parse(path).getroot().iter(f'{ALIGN_NS}Cell'):
            relation = cell.findtext(
                f'{ALIGN_NS}relation', default='='
            ).strip()
            e1 = cell.find(f'{ALIGN_NS}entity1').get(f'{RDF_NS}resource')
            e2 = cell.find(f'{ALIGN_NS}entity2').get(f'{RDF_NS}resource')
            if relation == '=':
                pairs.append((e1, e2))
    return pairs


def normalise_name(iri: str) -> str:
    return ''.join(parse_camel_plus(re.split('[#/]', iri)[-1]))


def query_vector(
    iri: str, mode: str, model, iri2name: Dict[str, str]
) -> Optional[np.ndarray]:
    iri_vec = (
        model.wv.get_vector(iri) if iri in model.wv.key_to_index else None
    )
    try:
        name_vec = embed_text(parse_camel_plus(iri2name[iri]), model)
    except ValueError:
        name_vec = None
    if mode == 'iri':
        return iri_vec
    if mode == 'name':
        return name_vec
    vecs = [
        v / np.linalg.norm(v) for v in (iri_vec, name_vec) if v is not None
    ]
    return np.mean(vecs, axis=0) if vecs else None


def summarise(ranks: List[int]) -> Dict[str, float]:
    ranks = np.array(ranks, dtype=float)
    out = {f'hits@{k}': float(np.mean(ranks <= k)) for k in HITS_AT}
    out['mrr'] = float(np.mean(1.0 / ranks))
    out['n'] = int(len(ranks))
    return out


def evaluate(model_path: str, ontology_dir: str, alignment_dir: str) -> Dict:
    model = gensim.models.Word2Vec.load(model_path)
    ontologies = load_ontologies(ontology_dir)
    onto_iri2embeds = embed_classes(ontologies, model, verbose=False)
    indices = build_indices(onto_iri2embeds, model.vector_size)

    # class IRI -> name of the index (ontology) holding it, and its local name
    iri2index, iri2name = {}, {}
    for onto in ontologies:
        for cls in onto.classes():
            iri2index[cls.iri] = onto.base_iri
            iri2name[cls.iri] = cls.name

    pairs = load_reference_pairs(alignment_dir)
    class_pairs = [
        (a, b) for a, b in pairs if a in iri2index and b in iri2index
    ]

    # ranks[mode][subset] -> list of ranks
    ranks = {m: defaultdict(list) for m in QUERY_MODES}
    for a, b in class_pairs:
        subset = (
            'same_name'
            if normalise_name(a) == normalise_name(b)
            else 'different_name'
        )
        for source, target in ((a, b), (b, a)):
            index = indices.get(iri2index[target])
            if index is None:
                continue
            for mode in QUERY_MODES:
                q = query_vector(source, mode, model, iri2name)
                if q is None:
                    rank = index.n_classes + 1  # cannot be found: worst rank
                else:
                    ranked = [iri for iri, _ in index.search(q, k=None)]
                    rank = (
                        ranked.index(target) + 1
                        if target in ranked
                        else index.n_classes + 1
                    )
                ranks[mode]['all'].append(rank)
                ranks[mode][subset].append(rank)

    return {
        'model': model_path,
        'ontologies': len(ontologies),
        'reference_pairs': len(pairs),
        'class_pairs_evaluated': len(class_pairs),
        'skipped_non_class_pairs': len(pairs) - len(class_pairs),
        'metrics': {
            m: {s: summarise(r) for s, r in subsets.items()}
            for m, subsets in ranks.items()
        },
    }


def print_report(result: Dict, previous: Optional[Dict]) -> None:
    print(f"\nModel: {result['model']}  |  ontologies: {result['ontologies']}")
    print(
        f"Reference pairs: {result['reference_pairs']}  |  "
        f"class pairs evaluated: {result['class_pairs_evaluated']} "
        f"(x2 directions)  |  skipped (properties etc.): "
        f"{result['skipped_non_class_pairs']}"
    )
    cols = [f'hits@{k}' for k in HITS_AT] + ['mrr']
    print(
        f"\n{'query':6} {'subset':15} {'n':>4} "
        + ' '.join(f'{c:>8}' for c in cols)
    )
    for mode in QUERY_MODES:
        for subset in ('all', 'same_name', 'different_name'):
            m = result['metrics'][mode].get(subset)
            if not m:
                continue
            line = f"{mode:6} {subset:15} {m['n']:>4} " + ' '.join(
                f"{m[c]:>8.3f}" for c in cols
            )
            if previous:
                p = previous['metrics'].get(mode, {}).get(subset)
                if p:
                    line += f"   (MRR {m['mrr'] - p['mrr']:+.3f} vs previous)"
            print(line)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--ontology_dir', default=ONTOLOGY_DIR)
    parser.add_argument(
        '--alignment_dir', default='./reference_alignments/conference'
    )
    parser.add_argument('--history', default='./output/eval_history.jsonl')
    parser.add_argument(
        '--label', default='', help='free-text note stored with this run'
    )
    args = parser.parse_args()

    result = evaluate(args.model, args.ontology_dir, args.alignment_dir)
    result['label'] = args.label
    result['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

    previous = None
    if os.path.exists(args.history):
        with open(args.history) as f:
            lines = [l for l in f if l.strip()]
        previous = json.loads(lines[-1]) if lines else None

    print_report(result, previous)

    os.makedirs(os.path.dirname(args.history) or '.', exist_ok=True)
    with open(args.history, 'a') as f:
        f.write(json.dumps(result) + '\n')
    print(f'\nRun appended to {args.history}')


if __name__ == '__main__':
    main()
