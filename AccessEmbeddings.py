import argparse
import configparser

import gensim
import numpy as np
from nltk import word_tokenize
from owlready2 import *


parser = argparse.ArgumentParser()
parser.add_argument("--ontology_file", type=str, default=None, help="The input ontology for embedding")
parser.add_argument("--embedding_dir", type=str, default=None, help="The output embedding directory")
parser.add_argument("--embedding_file", type=str, default=None, help="The output embedding file")
parser.add_argument("--config_file", type=str, default='default.cfg', help="Configuration file")
FLAGS, unparsed = parser.parse_known_args()

config = configparser.ConfigParser()
config.read(FLAGS.config_file)
if FLAGS.ontology_file is not None:
    config['BASIC']['ontology_file'] = FLAGS.ontology_file
if FLAGS.embedding_dir is not None:
    config['BASIC']['embedding_dir'] = FLAGS.embedding_dir
if FLAGS.embedding_file is not None:
    config['BASIC']['embedding_file'] = FLAGS.embedding_file

model = gensim.models.Word2Vec.load(config['BASIC']['embedding_dir'] + config['BASIC']['embedding_file'])
onto = get_ontology(config['BASIC']['ontology_file']).load()
classes = list(onto.classes())
c = classes[0]
c.iri in model.wv.index_to_key
iri_v = model.wv.get_vector(c.iri)
print(iri_v)

label = c.label[0]
text = ' '.join([re.sub(r'https?:\/\/.*[\r\n]*', '', w, flags=re.MULTILINE) for w in label.lower().split()])
words = [token.lower() for token in word_tokenize(text) if token.isalpha()]
n = 0
word_v = np.zeros(model.vector_size)
for word in words:
	if word not in model.wv.index_to_key:
		pass
	else:
		word_v += model.wv.get_vector(word)
		n += 1
word_v = word_v / n if n > 0 else word_v
print('\nWord vectors:\n', word_v)
