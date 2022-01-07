import argparse
import random
import gensim

import numpy as np
from nltk import word_tokenize
from owlready2 import *
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument('--train_subsumption_file', type=str, default='./foodon_subsume/train_subsumptions.csv')
parser.add_argument('--test_subsumption_file', type=str, default='./foodon_subsume/test_subsumptions.csv')
parser.add_argument('--train_onto_file', type=str, default='./foodon_subsume/foodon_train.owl')
parser.add_argument('--label_property', type=str, default='http://www.w3.org/2000/01/rdf-schema#label')
parser.add_argument('--train_pos_dup', type=int, default=2)
parser.add_argument('--train_neg_dup', type=int, default=2)
parser.add_argument('--use_contextual_candidates', type=bool, default=True)
parser.add_argument('--word2vec_embedding_file', type=str, default='./foodon_subsume/foodon.w2v.model')
parser.add_argument('--vector_type', type=str, default='word', help='iri, iri_word, word')
parser.add_argument('--classifier', type=str, default='lr', help='rf,mlp,lr')
FLAGS, unparsed = parser.parse_known_args()

start_time = datetime.datetime.now()

onto = get_ontology(FLAGS.train_onto_file).load()
named_classes = [c for c in onto.classes() if True not in c.deprecated and not c == owl.Thing]
iri_label = dict()
graph = default_world.as_rdflib_graph()
for p in FLAGS.label_property.split(','):
    q = 'SELECT ?s ?o WHERE {?s <%s> ?o.}' % p
    for res in list(graph.query(q)):
        iri = res[0].n3()[1:-1]
        lit = res[1]
        if hasattr(lit, 'language') and \
                (lit.language is None or lit.language == 'en' or lit.language == 'en-us' or lit.language == 'en-gb'):
            if iri in iri_label:
                if lit.value not in iri_label[iri]:
                    iri_label[iri].append(lit.value)
            else:
                iri_label[iri] = [lit.value]


def pre_process_label(input_label):
    text = ' '.join([re.sub(r'https?:\/\/.*[\r\n]*', '', w, flags=re.MULTILINE) for w in input_label.lower().split()])
    tokens = word_tokenize(text)
    return [token.lower() for token in tokens if token.isalpha()]


w2v = gensim.models.Word2Vec.load(FLAGS.word2vec_embedding_file)
iri_embedding = dict()
for c in onto.classes():
    if type(c) == entity.ThingClass and c is not owl.Thing and True not in c.deprecated:
        iri_vector = w2v.wv.get_vector(c.iri) if c.iri in w2v.wv.index_to_key else np.zeros(w2v.vector_size)
        word_vector = np.zeros(w2v.vector_size)
        n = 0
        if c.iri in iri_label:
            for lab in iri_label[c.iri]:
                words = pre_process_label(input_label=lab)
                for word in words:
                    if word in w2v.wv.index_to_key:
                        word_vector += w2v.wv.get_vector(word)
                        n += 1
        word_vector = word_vector / n if n > 0 else word_vector

        if FLAGS.vector_type == 'iri':
            iri_embedding[c.iri] = iri_vector
        elif FLAGS.vector_type == 'word':
            iri_embedding[c.iri] = word_vector
        elif FLAGS.vector_type == 'iri_word':
            iri_embedding[c.iri] = np.concatenate((iri_vector, word_vector))
        else:
            print('Unknown vector type')
            sys.exit(0)


read_subsumptions = lambda file_name: [line.strip().split(',') for line in open(file_name).readlines()]
train_subsumptions = read_subsumptions(FLAGS.train_subsumption_file)
neg_subsumptions = list()
for subs in train_subsumptions:
    c1 = subs[0]
    for neg_c in random.sample(set(named_classes) - IRIS[c1].ancestors(), FLAGS.train_neg_dup):
        neg_subsumptions.append([c1, neg_c.iri])
pos_subsumptions = FLAGS.train_pos_dup * train_subsumptions
print('Positive train subsumptions: %d' % len(pos_subsumptions))
print('Negative train subsumptions: %d' % len(neg_subsumptions))

subsumption_vector = lambda subsumption: np.concatenate((iri_embedding[subsumption[0]], iri_embedding[subsumption[1]]))
pos_X = [subsumption_vector(s) for s in pos_subsumptions]
pos_y = np.ones((len(pos_X)))
pos_X = np.array(pos_X)
neg_X = [subsumption_vector(s) for s in neg_subsumptions]
neg_y = np.zeros((len(neg_X)))
neg_X = np.array(neg_X)
X, y = np.concatenate((pos_X, neg_X)), np.concatenate((pos_y, neg_y))
X, y = shuffle(X, y, random_state=0)

if FLAGS.classifier == 'rf':
    model = RandomForestClassifier(n_estimators=100)
elif FLAGS.classifier == 'mlp':
    model = MLPClassifier(max_iter=1000, hidden_layer_sizes=200)
else:
    model = LogisticRegression(random_state=0)
model.fit(X, y)

end_time = datetime.datetime.now()
print('data pre-processing and training cost %.1f minutes' % ((end_time - start_time).seconds / 60))


start_time = datetime.datetime.now()
test_subsumptions = read_subsumptions(FLAGS.test_subsumption_file)

MRR_sum, hits1_sum, hits5_sum, hits10_sum = 0, 0, 0, 0
MRR, Hits1, Hits5, Hits10 = 0, 0, 0, 0
for k, test in enumerate(test_subsumptions):
    subcls, gt = test[0], test[1]
    if FLAGS.use_contextual_candidates:
        candidates = test[1:]
    else:
        candidates = [c.iri for c in set(named_classes) - IRIS[subcls].ancestors()]
        if gt not in candidates:
            candidates.append(gt)

    candidate_subsumptions = [[subcls, c] for c in candidates]
    candidate_scores = np.zeros(len(candidate_subsumptions))

    V = np.array([subsumption_vector(candidate_subsumption) for candidate_subsumption in candidate_subsumptions])
    P = model.predict_proba(V)[:, 1]
    sorted_indexes = np.argsort(P)[::-1]
    sorted_classes = list()
    for j in sorted_indexes:
        sorted_classes.append(candidates[j])
    rank = sorted_classes.index(gt) + 1

    MRR_sum += 1.0 / rank
    hits1_sum += 1 if gt in sorted_classes[:1] else 0
    hits5_sum += 1 if gt in sorted_classes[:5] else 0
    hits10_sum += 1 if gt in sorted_classes[:10] else 0
    num = k + 1
    MRR, Hits1, Hits5, Hits10 = MRR_sum / num, hits1_sum / num, hits5_sum / num, hits10_sum / num
    if num % 500 == 0:
        print('\n%d tested, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n' % (num, MRR, Hits1, Hits5, Hits10))
print('\nAll tested, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n' % (MRR, Hits1, Hits5, Hits10))
end_time = datetime.datetime.now()
print('Evaluation costs %.1f minutes' % ((end_time - start_time).seconds / 60))
