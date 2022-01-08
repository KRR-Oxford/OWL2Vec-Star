import os
import rdflib
import argparse
from owlready2 import *

parser = argparse.ArgumentParser()
parser.add_argument('--onto_file', type=str, default='foodon_subsume/foodon_train.owl')
parser.add_argument('--out_dir', type=str, default='foodon_subsume/openke/')
FLAGS, unparsed = parser.parse_known_args()

if not os.path.exists(FLAGS.out_dir):
    os.mkdir(FLAGS.out_dir)

entities, relations, triples = set(), set(), list()
g = rdflib.Graph()
g.parse(FLAGS.onto_file)
for (s, p, o) in g:
    s, p, o = str(s), str(p), str(o)
    s = s.strip().replace('\t', ' ').replace('\n', '')
    p = p.strip().replace('\t', ' ').replace('\n', '')
    o = o.strip().replace('\t', ' ').replace('\n', '')
    if s == '' or p == '' or o == '':
        continue
    triples.append([s, p, o])
    entities.add(s)
    entities.add(o)
    relations.add(p)
entities, relations = list(entities), list(relations)
print('entities: %d, relations: %d, triples: %d' % (len(entities), len(relations), len(triples)))

entity_id, relation_id = dict(), dict()
for i, entity in enumerate(entities):
    entity_id[entity] = i
for i, relation in enumerate(relations):
    relation_id[relation] = i

with open(os.path.join(FLAGS.out_dir, 'entity2id.txt'), 'w') as f:
    f.write('%d\n' % len(entities))
    for i, entity in enumerate(entities):
        f.write('%s\t%d\n' % (entity, i))

with open(os.path.join(FLAGS.out_dir, 'relation2id.txt'), 'w') as f:
    f.write('%d\n' % len(relations))
    for i, relation in enumerate(relations):
        f.write('%s\t%d\n' % (relation, i))

with open(os.path.join(FLAGS.out_dir, 'train2id.txt'), 'w') as f:
    f.write('%d\n' % len(triples))
    for i, triple in enumerate(triples):
        sub_id = entity_id[triple[0]]
        rel_id = relation_id[triple[1]]
        obj_id = entity_id[triple[2]]
        f.write('%d %d %d\n' % (sub_id, obj_id, rel_id))
