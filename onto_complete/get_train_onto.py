# generate the ontology for training (i.e., the original ontology - test and valid subsumptions)

import argparse
from owlready2 import *

parser = argparse.ArgumentParser()
parser.add_argument('--onto_file', type=str,
                    default='foodon-merged.0.4.8.owl')
parser.add_argument('--valid_subsumption_file', type=str, default='./foodon_subsume/valid_subsumptions.csv')
parser.add_argument('--test_subsumption_file', type=str, default='./foodon_subsume/test_subsumptions.csv')
parser.add_argument('--train_onto_file', type=str, default='./foodon_subsume/foodon_train.owl')
FLAGS, unparsed = parser.parse_known_args()

read_subsumptions = lambda file_name: [line.strip().split(',') for line in open(file_name).readlines()]
test_subsumptions = read_subsumptions(FLAGS.test_subsumption_file)
valid_subsumptions = read_subsumptions(FLAGS.valid_subsumption_file)

onto = get_ontology(FLAGS.onto_file).load()

for subsumptions in valid_subsumptions + test_subsumptions:
    subc, supc = IRIS[subsumptions[0]], IRIS[subsumptions[1]]
    if supc in subc.is_a:
        subc.is_a.remove(supc)
    else:
        print('wrong subsumption: %s, %s' % (subc, supc))

onto.save(file=FLAGS.train_onto_file, format='rdfxml')
print('train ontology saved!')
