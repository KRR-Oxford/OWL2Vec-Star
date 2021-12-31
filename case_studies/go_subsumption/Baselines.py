import numpy as np
import pandas as pd
import gensim
import multiprocessing
import argparse
import random
import json
import sys

from owl2vec_star.lib.Evaluator import Evaluator
from owl2vec_star.lib.RDF2Vec_Embed import get_rdf2vec_embed

parser = argparse.ArgumentParser(description="The is to evaluate baselines such as RDF2Vec.")
parser.add_argument("--onto_file", type=str, default="go.train.owl")
parser.add_argument("--train_file", type=str, default="train.csv")
parser.add_argument("--valid_file", type=str, default="valid.csv")
parser.add_argument("--test_file", type=str, default="test.csv")
parser.add_argument("--class_file", type=str, default="classes.txt")
parser.add_argument("--inferred_ancestor_file", type=str, default="inferred_ancestors.txt")
# el embedding file
parser.add_argument("--el_embedding_file", type=str, default='cls_embeddings.pkl', help="class embedding file")
parser.add_argument("--transe_embedding_file", type=str, default='openke_go/transe_embedding.vec.json')
parser.add_argument("--transr_embedding_file", type=str, default='openke_go/transr_embedding.vec.json')
parser.add_argument("--distmult_embedding_file", type=str, default='openke_go/distmult_embedding.vec.json')
parser.add_argument("--openke_entity_file", type=str, default='openke_go/entity2id.txt')

parser.add_argument("--embedding_type", type=str, default='rdf2vec',
                    help='rdf2vec, opa2vec, el, onto2vec, transe, distmult, transr')
parser.add_argument("--embedsize", type=int, default=100, help="Embedding size of word2vec")
parser.add_argument("--input_type", type=str, default="concatenate", help='concatenate, minus')

# RDF2Vec hyper parameters
parser.add_argument("--walk_depth", type=int, default=2)
parser.add_argument("--walker", type=str, default="wl", help="random, wl")

# OPA2Vec hyper parameters
parser.add_argument("--axiom_file", type=str, default='axioms.txt', help="axioms.txt or axioms_hermit.txt")
parser.add_argument("--annotation_file", type=str, default='annotations.txt', help="literal axioms")
parser.add_argument("--windsize", type=int, default=5, help="Window size for word2vec model")
parser.add_argument("--mincount", type=int, default=0, help="Minimum count value for word2vec model")
parser.add_argument("--model", type=str, default='sg', help="word2vec architecture: sg or cbow")
parser.add_argument("--pretrained", type=str, default="none",
                    help="/Users/jiahen/Data/w2v_model/enwiki_model/word2vec_gensim or none")

FLAGS, unparsed = parser.parse_known_args()

print("\n		1.learn embedding ... \n")

classes = [line.strip() for line in open(FLAGS.class_file).readlines()]
candidate_num = len(classes)

if FLAGS.embedding_type.lower() == 'rdf2vec':
    classes_e = get_rdf2vec_embed(onto_file=FLAGS.onto_file, walker_type=FLAGS.walker,
                                  walk_depth=FLAGS.walk_depth, embed_size=FLAGS.embedsize,
                                  classes=classes)

elif FLAGS.embedding_type.lower() in ['opa2vec', 'onto2vec']:
    if FLAGS.embedding_type.lower() == 'opa2vec':
        lines = open(FLAGS.axiom_file).readlines() + open(FLAGS.annotation_file).readlines()
    else:
        lines = open(FLAGS.axiom_file).readlines()
    sentences = list()
    for line in lines:
        sentence = [item.strip().lower() for item in line.strip().split()]
        sentences.append(sentence)
    if FLAGS.pretrained.lower() == 'none' or FLAGS.pretrained == '':
        sg_v = 1 if FLAGS.model == 'sg' else 0
        w2v = gensim.models.Word2Vec(sentences, sg=sg_v, min_count=FLAGS.mincount, vector_size=FLAGS.embedsize,
                                     window=FLAGS.windsize, workers=multiprocessing.cpu_count())
    else:
        w2v = gensim.models.Word2Vec.load(FLAGS.pretrained)
        w2v.min_count = FLAGS.mincount
        w2v.build_vocab(sentences, update=True)
        w2v.train(sentences, total_examples=w2v.corpus_count, epochs=100)

    classes_e = [w2v.wv.get_vector(c.lower()) if c.lower() in w2v.wv.index_to_key else np.zeros(w2v.vector_size)
                 for c in classes]
    classes_e = np.array(classes_e)

elif FLAGS.embedding_type.lower() == 'el':
    cls_embeddings = pd.read_pickle(FLAGS.el_embedding_file)
    embedding_classes = list(cls_embeddings['classes'])
    embeddings = list(cls_embeddings['embeddings'])
    FLAGS.embedsize = embeddings[0].shape[0]
    classes_e = list()
    for c in classes:
        c_name = '<' + c + '>'
        if c_name in embedding_classes:
            c_index = embedding_classes.index(c_name)
            classes_e.append(embeddings[c_index])
        else:
            classes_e.append(np.zeros(FLAGS.embedsize))
    classes_e = np.array(classes_e)

elif FLAGS.embedding_type.lower() in ['transe', 'distmult', 'transr']:
    if FLAGS.embedding_type.lower() == 'transe':
        embed_f = open(FLAGS.transe_embedding_file, 'r')
    elif FLAGS.embedding_type.lower() == 'distmult':
        embed_f = open(FLAGS.distmult_embedding_file, 'r')
    else:
        embed_f = open(FLAGS.transr_embedding_file, 'r')
    embeddings = json.loads(embed_f.read())
    ent_embeddings = embeddings['ent_embeddings']
    entity_id = dict()
    with open(FLAGS.openke_entity_file) as f:
        for line in f.readlines()[1:]:
            tmp = line.strip().split('\t')
            entity_id[tmp[0]] = int(tmp[1])

    FLAGS.embedsize = len(ent_embeddings[0])
    classes_e = list()
    for c in classes:
        cid = entity_id[c]
        classes_e.append(np.array(ent_embeddings[cid]))
    classes_e = np.array(classes_e)

else:
    print('%s: embedding type not implemented' % FLAGS.embedding_type)
    sys.exit(0)

print("\n		2.sample ... \n")
train_samples = [line.strip().split(',') for line in open(FLAGS.train_file).readlines()]
valid_samples = [line.strip().split(',') for line in open(FLAGS.valid_file).readlines()]
test_samples = [line.strip().split(',') for line in open(FLAGS.test_file).readlines()]
random.shuffle(train_samples)

train_x_list, train_y_list = list(), list()
for s in train_samples:
    sub, sup, label = s[0], s[1], s[2]
    sub_v = classes_e[classes.index(sub)]
    sup_v = classes_e[classes.index(sup)]
    if not (np.all(sub_v == 0) or np.all(sup_v == 0)):
        if FLAGS.input_type == 'concatenate':
            train_x_list.append(np.concatenate((sub_v, sup_v)))
        else:
            train_x_list.append(sub_v - sup_v)
        train_y_list.append(int(label))
train_X, train_y = np.array(train_x_list), np.array(train_y_list)
print('train_X: %s, train_y: %s' % (str(train_X.shape), str(train_y.shape)))

inferred_ancestors = dict()
with open(FLAGS.inferred_ancestor_file) as f:
    for line in f.readlines():
        all_infer_classes = line.strip().split(',')
        cls = all_infer_classes[0]
        inferred_ancestors[cls] = all_infer_classes


class InclusionEvaluator(Evaluator):
    def __init__(self, valid_samples, test_samples, train_X, train_y):
        super(InclusionEvaluator, self).__init__(valid_samples, test_samples, train_X, train_y)

    def evaluate(self, model, eva_samples):
        MRR_sum, hits1_sum, hits5_sum, hits10_sum = 0, 0, 0, 0
        for sample in eva_samples:
            sub, gt = sample[0], sample[1]
            sub_index = classes.index(sub)
            sub_v = classes_e[sub_index]
            if FLAGS.input_type == 'concatenate':
                X = np.concatenate((np.array([sub_v] * candidate_num), classes_e), axis=1)
            else:
                X = np.array([sub_v] * candidate_num) - classes_e
            P = model.predict_proba(X)[:, 1]
            sorted_indexes = np.argsort(P)[::-1]
            sorted_classes = list()
            for j in sorted_indexes:
                if classes[j] not in inferred_ancestors[sub]:
                    sorted_classes.append(classes[j])
            rank = sorted_classes.index(gt) + 1
            MRR_sum += 1.0 / rank
            hits1_sum += 1 if gt in sorted_classes[:1] else 0
            hits5_sum += 1 if gt in sorted_classes[:5] else 0
            hits10_sum += 1 if gt in sorted_classes[:10] else 0
        eva_n = len(eva_samples)
        e_MRR, hits1, hits5, hits10 = MRR_sum / eva_n, hits1_sum / eva_n, hits5_sum / eva_n, hits10_sum / eva_n
        return e_MRR, hits1, hits5, hits10


print("\n		3.Train, valid and test ... \n")
evaluator = InclusionEvaluator(valid_samples, test_samples, train_X, train_y)
evaluator.run_random_forest()
