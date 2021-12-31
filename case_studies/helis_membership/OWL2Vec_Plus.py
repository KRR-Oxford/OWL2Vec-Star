import numpy as np
import argparse
import random
import re
import multiprocessing
import gensim
import sys
from nltk import word_tokenize

from owl2vec_star.lib.Evaluator import Evaluator
from owl2vec_star.lib.RDF2Vec_Embed import get_rdf2vec_walks

parser = argparse.ArgumentParser(description="The is to evaluate baselines.")
parser.add_argument("--onto_file", type=str, default="helis_v1.00.train.projection.ttl",
                    help='helis_v1.00.train.owl or helis_v1.00.train.projection.ttl or helis_v1.00.train.projection.r.ttl')
parser.add_argument("--train_file", type=str, default="train.csv")
parser.add_argument("--valid_file", type=str, default="valid.csv")
parser.add_argument("--test_file", type=str, default="test.csv")
parser.add_argument("--class_file", type=str, default="classes.txt")
parser.add_argument("--individual_file", type=str, default="individuals.txt")
parser.add_argument("--inferred_class_file", type=str, default="inferred_classes.txt")

# hyper parameters
parser.add_argument("--embedsize", type=int, default=100, help="Embedding size of word2vec")
parser.add_argument("--URI_Doc", type=str, default="yes")
parser.add_argument("--Lit_Doc", type=str, default="no")
parser.add_argument("--Mix_Doc", type=str, default="no")
parser.add_argument("--Mix_Type", type=str, default="random", help="random, all")
parser.add_argument("--Embed_Out_URI", type=str, default="yes")
parser.add_argument("--Embed_Out_Words", type=str, default="no")
parser.add_argument("--input_type", type=str, default="concatenate", help='concatenate, minus')

parser.add_argument("--walk_depth", type=int, default=2)
parser.add_argument("--walker", type=str, default="random", help="random, wl")
parser.add_argument("--axiom_file", type=str, default='axioms.txt', help="Corpus of Axioms")
parser.add_argument("--annotation_file", type=str, default='annotations.txt', help="Corpus of Literals")

parser.add_argument("--pretrained", type=str, default="none",
                    help="~/Data/w2v_model/enwiki_model/word2vec_gensim or none")

FLAGS, unparsed = parser.parse_known_args()

print(FLAGS.walker)
print(FLAGS.walk_depth)
if FLAGS.Embed_Out_Words.lower() == 'yes' and FLAGS.Mix_Doc.lower() == 'no' and \
        FLAGS.Lit_Doc.lower() == 'no' and FLAGS.pretrained == 'none':
    print('Can not embed words with no Lit Doc or Mix Doc or pretrained model')
    sys.exit(0)


def embed(model, instances):
    def word_embeding(inst):
        v = np.zeros(model.vector_size)
        if inst in uri_label:
            words = uri_label.get(inst)
            n = 0
            for word in words:
                if word in model.wv.index_to_key:
                    v += model.wv.get_vector(word)
                    n += 1
            return v / n if n > 0 else v
        else:
            return v

    feature_vectors = []
    for instance in instances:
        if FLAGS.Embed_Out_Words.lower() == 'yes' and FLAGS.Embed_Out_URI.lower() == 'yes':
            v_uri = model.wv.get_vector(instance) if instance in model.wv.index_to_key else np.zeros(model.vector_size)
            v_word = word_embeding(inst=instance)
            feature_vectors.append(np.concatenate((v_uri, v_word)))

        elif FLAGS.Embed_Out_Words.lower() == 'no' and FLAGS.Embed_Out_URI.lower() == 'yes':
            v_uri = model.wv.get_vector(instance) if instance in model.wv.index_to_key else np.zeros(model.vector_size)
            feature_vectors.append(v_uri)

        elif FLAGS.Embed_Out_Words.lower() == 'yes' and FLAGS.Embed_Out_URI.lower() == 'no':
            v_word = word_embeding(inst=instance)
            feature_vectors.append(v_word)

        else:
            print("Unknown embed out type")
            sys.exit(0)

    return feature_vectors


def pre_process_words(words):
    text = ' '.join([re.sub(r'https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) for word in words])
    tokens = word_tokenize(text)
    processed_tokens = [token.lower() for token in tokens if token.isalpha()]
    return processed_tokens


def URI_parse(uri):
    """Parse a URI: remove the prefix, parse the name part (Camel cases are plit)"""
    uri = re.sub("http[a-zA-Z0-9:/._-]+#", "", uri)
    uri = uri.replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('/', ' '). \
        replace('"', ' ').replace("'", ' ')
    words = []
    for item in uri.split():
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', item)
        for m in matches:
            word = m.group(0)
            if word.isalpha():
                words.append(word.lower())
    return words


print("\n		1.Extract corpus and learning embedding ... \n")
classes = [line.strip() for line in open(FLAGS.class_file).readlines()]
candidate_num = len(classes)
individuals = [line.strip() for line in open(FLAGS.individual_file).readlines()]

annotations = list()
uri_label = dict()
for line in open(FLAGS.annotation_file).readlines():
    tmp = line.strip().split()
    if tmp[1] not in ['http://www.w3.org/2000/01/rdf-schema#label', 'http://www.fbk.eu/ontologies/virtualcoach#id'] \
            and tmp[0] in classes + individuals:
        annotations.append(tmp)
    if tmp[1] == 'http://www.w3.org/2000/01/rdf-schema#label':
        uri_label[tmp[0]] = pre_process_words(tmp[2:])
for ent in individuals + classes:
    if ent not in uri_label:
        uri_label[ent] = URI_parse(ent)

walk_sentences, axiom_sentences = list(), list()
if FLAGS.URI_Doc.lower() == 'yes':
    walks_ = get_rdf2vec_walks(onto_file=FLAGS.onto_file, walker_type=FLAGS.walker,
                               walk_depth=FLAGS.walk_depth, classes=classes + individuals)
    print('Extracted {} walks for {} classes/individuals!'.format(len(walks_), len(classes) + len(individuals)))
    walk_sentences += [list(map(str, x)) for x in walks_]
    for line in open(FLAGS.axiom_file).readlines():
        axiom_sentence = [item for item in line.strip().split()]
        axiom_sentences.append(axiom_sentence)
    print('Extracted %d axiom sentences' % len(axiom_sentences))
URI_Doc = walk_sentences + axiom_sentences

Lit_Doc = list()
if FLAGS.Lit_Doc.lower() == 'yes':
    for annotation in annotations:
        processed_words = pre_process_words(annotation[2:])
        if len(processed_words) > 0:
            Lit_Doc.append(uri_label[annotation[0]] + processed_words)
    print('Extracted %d literal annotations' % len(Lit_Doc))

    for sentence in walk_sentences:
        lit_sentence = list()
        for item in sentence:
            if item in uri_label:
                lit_sentence += uri_label[item]
            elif item.startswith('http://www.w3.org'):
                lit_sentence += [item.split('#')[1].lower()]
            else:
                lit_sentence += [item]
        Lit_Doc.append(lit_sentence)

    for sentence in axiom_sentences:
        lit_sentence = list()
        for item in sentence:
            lit_sentence += uri_label[item] if item in uri_label else [item.lower()]
        Lit_Doc.append(lit_sentence)

Mix_Doc = list()
if FLAGS.Mix_Doc.lower() == 'yes':
    for sentence in walk_sentences:
        if FLAGS.Mix_Type.lower() == 'all':
            for index in range(len(sentence)):
                mix_sentence = list()
                for i, item in enumerate(sentence):
                    if i == index:
                        mix_sentence += [item]
                    else:
                        if item in uri_label:
                            mix_sentence += uri_label[item]
                        elif item.startswith('http://www.w3.org'):
                            mix_sentence += [item.split('#')[1].lower()]
                        else:
                            mix_sentence += [item]
                Mix_Doc.append(mix_sentence)
        elif FLAGS.Mix_Type.lower() == 'random':
            random_index = random.randint(0, len(sentence) - 1)
            mix_sentence = list()
            for i, item in enumerate(sentence):
                if i == random_index:
                    mix_sentence += [item]
                else:
                    if item in uri_label:
                        mix_sentence += uri_label[item]
                    elif item.startswith('http://www.w3.org'):
                        mix_sentence += [item.split('#')[1].lower()]
                    else:
                        mix_sentence += [item]
            Mix_Doc.append(mix_sentence)

    for sentence in axiom_sentences:
        if FLAGS.Mix_Type.lower() == 'all':
            for index in range(len(sentence)):
                random_index = random.randint(0, len(sentence) - 1)
                mix_sentence = list()
                for i, item in enumerate(sentence):
                    if i == random_index:
                        mix_sentence += [item]
                    else:
                        mix_sentence += uri_label[item] if item in uri_label else [item.lower()]
                Mix_Doc.append(mix_sentence)
        elif FLAGS.Mix_Type.lower() == 'random':
            random_index = random.randint(0, len(sentence) - 1)
            mix_sentence = list()
            for i, item in enumerate(sentence):
                if i == random_index:
                    mix_sentence += [item]
                else:
                    mix_sentence += uri_label[item] if item in uri_label else [item.lower()]
            Mix_Doc.append(mix_sentence)

print('URI_Doc: %d, Lit_Doc: %d, Mix_Doc: %d' % (len(URI_Doc), len(Lit_Doc), len(Mix_Doc)))
all_doc = URI_Doc + Lit_Doc + Mix_Doc
random.shuffle(all_doc)

# learn the embeddings
if FLAGS.pretrained.lower() == 'none' or FLAGS.pretrained == '':
    model_ = gensim.models.Word2Vec(all_doc, vector_size=FLAGS.embedsize, window=5, workers=multiprocessing.cpu_count(),
                                    sg=1, epochs=10, negative=25, min_count=1, seed=42)
else:
    model_ = gensim.models.Word2Vec.load(FLAGS.pretrained)
    if len(all_doc) > 0:
        model_.min_count = 1
        model_.build_vocab(all_doc, update=True)
        model_.train(all_doc, total_examples=model_.corpus_count, epochs=100)

classes_e = embed(model=model_, instances=classes)
new_embedsize = classes_e[0].shape[0]
individuals_e = embed(model=model_, instances=individuals)

print("\n		2.Train and test ... \n")
train_samples = [line.strip().split(',') for line in open(FLAGS.train_file).readlines()]
valid_samples = [line.strip().split(',') for line in open(FLAGS.valid_file).readlines()]
test_samples = [line.strip().split(',') for line in open(FLAGS.test_file).readlines()]
random.shuffle(train_samples)

train_x_list, train_y_list = list(), list()
for s in train_samples:
    ind, cls, label = s[0], s[1], s[2]
    ind_v = individuals_e[individuals.index(ind)]
    cls_v = classes_e[classes.index(cls)]
    if not (np.all(ind_v == 0) or np.all(cls_v == 0)):
        if FLAGS.input_type == 'concatenate':
            train_x_list.append(np.concatenate((ind_v, cls_v)))
        else:
            train_x_list.append(ind_v - cls_v)
        train_y_list.append(int(label))
train_X, train_y = np.array(train_x_list), np.array(train_y_list)
print('train_X: %s, train_y: %s' % (str(train_X.shape), str(train_y.shape)))

ind_infer_classes = dict()
with open(FLAGS.inferred_class_file) as f:
    for line in f.readlines():
        tmp = line.strip().split(',')
        ind = tmp[0]
        ind_infer_classes[ind] = tmp[1:]


class InclusionEvaluator(Evaluator):
    def __init__(self, valid_samples, test_samples, train_X, train_y):
        super(InclusionEvaluator, self).__init__(valid_samples, test_samples, train_X, train_y)

    def evaluate(self, model, eva_samples):
        MRR_sum, hits1_sum, hits5_sum, hits10_sum = 0, 0, 0, 0
        random.shuffle(eva_samples)
        for k, sample in enumerate(eva_samples):
            individual, gt = sample[0], sample[1]
            individual_index = individuals.index(individual)
            individual_v = individuals_e[individual_index]
            if FLAGS.input_type == 'concatenate':
                X = np.concatenate((np.array([individual_v] * candidate_num), classes_e), axis=1)
            else:
                X = np.array([individual_v] * candidate_num) - classes_e
            P = model.predict_proba(X)[:, 1]
            sorted_indexes = np.argsort(P)[::-1]
            sorted_classes = list()
            for j in sorted_indexes:
                if classes[j] not in ind_infer_classes[individual]:
                    sorted_classes.append(classes[j])
            rank = sorted_classes.index(gt) + 1
            MRR_sum += 1.0 / rank
            hits1_sum += 1 if gt in sorted_classes[:1] else 0
            hits5_sum += 1 if gt in sorted_classes[:5] else 0
            hits10_sum += 1 if gt in sorted_classes[:10] else 0
            num = k + 1
            if num % 5 == 0:
                print('\n%d tested, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n' %
                      (num, MRR_sum / num, hits1_sum / num, hits5_sum / num, hits10_sum / num))
        eva_n = len(eva_samples)
        e_MRR, hits1, hits5, hits10 = MRR_sum / eva_n, hits1_sum / eva_n, hits5_sum / eva_n, hits10_sum / eva_n
        return e_MRR, hits1, hits5, hits10


print("\n		2.Train and test ... \n")
evaluator = InclusionEvaluator(valid_samples, test_samples, train_X, train_y)
evaluator.run_random_forest()
# evaluator.run_mlp()
# evaluator.run_logistic_regression()
# evaluator.run_svm()
# evaluator.run_decision_tree()
# evaluator.run_linear_svc()
# evaluator.run_sgd_log()
