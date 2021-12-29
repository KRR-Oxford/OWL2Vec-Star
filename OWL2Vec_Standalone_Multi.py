import os
import time
import argparse
import random
import multiprocessing
import gensim
import configparser

from owl2vec_star.lib.RDF2Vec_Embed import get_rdf2vec_walks
from owl2vec_star.lib.Label import pre_process_words, URI_parse
from owl2vec_star.lib.Onto_Projection import Reasoner, OntologyProjection

parser = argparse.ArgumentParser()
parser.add_argument("--ontology_dir", type=str, default=None, help="The directory of input ontologies for embedding")
parser.add_argument("--embedding_dir", type=str, default=None, help="The output embedding directory")
parser.add_argument("--config_file", type=str, default='default_multi.cfg', help="Configuration file")
parser.add_argument("--URI_Doc", help="Using URI document", action="store_true")
parser.add_argument("--Lit_Doc", help="Using literal document", action="store_true")
parser.add_argument("--Mix_Doc", help="Using mixture document", action="store_true")
FLAGS, unparsed = parser.parse_known_args()

# read and combine configurations
# overwrite the parameters in the configuration file by the command parameters
config = configparser.ConfigParser()
config.read(FLAGS.config_file)
if FLAGS.ontology_dir is not None:
    config['BASIC']['ontology_dir'] = FLAGS.ontology_dir
if FLAGS.embedding_dir is not None:
    config['BASIC']['embedding_dir'] = FLAGS.embedding_dir
if FLAGS.URI_Doc:
    config['DOCUMENT']['URI_Doc'] = 'yes'
if FLAGS.Lit_Doc:
    config['DOCUMENT']['Lit_Doc'] = 'yes'
if FLAGS.Mix_Doc:
    config['DOCUMENT']['Mix_Doc'] = 'yes'
if 'cache_dir' not in config['DOCUMENT']:
    config['DOCUMENT']['cache_dir'] = './cache'
if not os.path.exists(config['DOCUMENT']['cache_dir']):
    os.mkdir(config['DOCUMENT']['cache_dir'])
if 'embedding_dir' not in config['BASIC']:
    config['BASIC']['embedding_dir'] = os.path.join(config['DOCUMENT']['cache_dir'], 'output')

start_time = time.time()

walk_sentences, axiom_sentences = list(), list()
uri_label, annotations = dict(), list()
for file_name in os.listdir(config['BASIC']['ontology_dir']):
    if not file_name.endswith('.owl'):
        continue
    ONTO_FILE = os.path.join(config['BASIC']['ontology_dir'], file_name)
    print('\nProcessing %s' % file_name)
    projection = OntologyProjection(ONTO_FILE, reasoner=Reasoner.STRUCTURAL, only_taxonomy=False,
                                    bidirectional_taxonomy=True, include_literals=True, avoid_properties=set(),
                                    additional_preferred_labels_annotations=set(),
                                    additional_synonyms_annotations=set(), memory_reasoner='13351')

    # Extract and save seed entities (classes and individuals)
    print('... Extract entities (classes and individuals) ...')
    projection.extractEntityURIs()
    classes = projection.getClassURIs()
    individuals = projection.getIndividualURIs()
    entities = classes.union(individuals)
    with open(os.path.join(config['DOCUMENT']['cache_dir'], 'entities.txt'), 'a') as f:
        for e in entities:
            f.write('%s\n' % e)

    # Extract and save axioms in Manchester Syntax
    print('... Extract axioms ...')
    projection.createManchesterSyntaxAxioms()
    with open(os.path.join(config['DOCUMENT']['cache_dir'], 'axioms.txt'), 'a') as f:
        for ax in projection.axioms_manchester:
            axiom_sentence = [item for item in ax.split()]
            axiom_sentences.append(axiom_sentence)
            f.write('%s\n' % ax)
    print('... %d axioms ...' % len(axiom_sentences))

    # Read annotations including rdfs:label and other literals from the ontology
    #   Extract annotations: 1) English label of each entity, by rdfs:label or skos:preferredLabel
    #                        2) None label annotations as sentences of the literal document
    print('... Extract annotations ...')
    projection.indexAnnotations()
    with open(os.path.join(config['DOCUMENT']['cache_dir'], 'annotations.txt'), 'a') as f:
        for e in entities:
            if e in projection.entityToPreferredLabels and len(projection.entityToPreferredLabels[e]) > 0:
                label = list(projection.entityToPreferredLabels[e])[0]
                v = pre_process_words(words=label.split())
                uri_label[e] = v
                f.write('%s preferred_label %s\n' % (e, v))
        for e in entities:
            if e in projection.entityToAllLexicalLabels:
                for v in projection.entityToAllLexicalLabels[e]:
                    if (v is not None) and \
                            (not (e in projection.entityToPreferredLabels and v in projection.entityToPreferredLabels[e])):
                        annotation = [e] + v.split()
                        annotations.append(annotation)
                        f.write('%s\n' % ' '.join(annotation))

    # project ontology to RDF graph (optionally) and extract walks
    if 'ontology_projection' in config['DOCUMENT'] and config['DOCUMENT']['ontology_projection'] == 'yes':
        print('... Calculate the ontology projection ...')
        projection.extractProjection()
        onto_projection_file = os.path.join(config['DOCUMENT']['cache_dir'], 'projection.ttl')
        projection.saveProjectionGraph(onto_projection_file)
        ONTO_FILE = onto_projection_file
    print('... Generate walks ...')
    walks_ = get_rdf2vec_walks(onto_file=ONTO_FILE, walker_type=config['DOCUMENT']['walker'],
                               walk_depth=int(config['DOCUMENT']['walk_depth']), classes=entities)
    print('... %d walks for %d seed entities ...' % (len(walks_), len(entities)))
    walk_sentences += [list(map(str, x)) for x in walks_]

# collect URI documents
# two parts: axiom sentences + walk sentences
URI_Doc = list()
if 'URI_Doc' in config['DOCUMENT'] and config['DOCUMENT']['URI_Doc'] == 'yes':
    print('Extracted %d axiom sentences' % len(axiom_sentences))
    URI_Doc = walk_sentences + axiom_sentences


# Some entities have English labels
# Keep the name of built-in properties (those starting with http://www.w3.org)
# Some entities have no labels, then use the words in their URI name
def label_item(item):
    if item in uri_label:
        return uri_label[item]
    elif item.startswith('http://www.w3.org'):
        return [item.split('#')[1].lower()]
    elif item.startswith('http://'):
        return URI_parse(uri=item)
    else:
        # return [item.lower()]
        return ''


# read literal document
# two parts: literals in the annotations (subject's label + literal words)
#            replacing walk/axiom sentences by words in their labels
Lit_Doc = list()
if 'Lit_Doc' in config['DOCUMENT'] and config['DOCUMENT']['Lit_Doc'] == 'yes':
    print('\n\nGenerate literal document')
    for annotation in annotations:
        processed_words = pre_process_words(annotation[1:])
        if len(processed_words) > 0:
            Lit_Doc.append(label_item(item=annotation[0]) + processed_words)
    print('... Extracted %d annotation sentences ...' % len(Lit_Doc))

    for sentence in walk_sentences + axiom_sentences:
        lit_sentence = list()
        for item in sentence:
            lit_sentence += label_item(item=item)
        Lit_Doc.append(lit_sentence)

# for each axiom/walk sentence, generate mixture sentence(s) by two strategies:
#   all): for each entity, keep its entity URI, replace the others by label words
#   random): randomly select one entity, keep its entity URI, replace the others by label words
Mix_Doc = list()
if 'Mix_Doc' in config['DOCUMENT'] and config['DOCUMENT']['Mix_Doc'] == 'yes':
    print('\n\nGenerate mixture document')
    for sentence in walk_sentences + axiom_sentences:
        if config['DOCUMENT']['Mix_Type'] == 'all':
            for index in range(len(sentence)):
                mix_sentence = list()
                for i, item in enumerate(sentence):
                    mix_sentence += [item] if i == index else label_item(item=item)
                Mix_Doc.append(mix_sentence)
        elif config['DOCUMENT']['Mix_Type'] == 'random':
            random_index = random.randint(0, len(sentence) - 1)
            mix_sentence = list()
            for i, item in enumerate(sentence):
                mix_sentence += [item] if i == random_index else label_item(item=item)
            Mix_Doc.append(mix_sentence)

print('\n\nURI_Doc: %d, Lit_Doc: %d, Mix_Doc: %d' % (len(URI_Doc), len(Lit_Doc), len(Mix_Doc)))
all_doc = URI_Doc + Lit_Doc + Mix_Doc
print('Time for document construction: %s seconds' % (time.time() - start_time))
random.shuffle(all_doc)

# learn the embedding model (train a new model or fine tune the pre-trained model)
start_time = time.time()
if 'pre_train_model' not in config['MODEL'] or not os.path.exists(config['MODEL']['pre_train_model']):
    print('\n\nTrain the embedding model')
    model_ = gensim.models.Word2Vec(all_doc, vector_size=int(config['MODEL']['embed_size']),
                                    window=int(config['MODEL']['window']),
                                    workers=multiprocessing.cpu_count(),
                                    sg=1, epochs=int(config['MODEL']['iteration']),
                                    negative=int(config['MODEL']['negative']),
                                    min_count=int(config['MODEL']['min_count']), seed=int(config['MODEL']['seed']))
else:
    print('\n\nFine-tune the pre-trained embedding model')
    model_ = gensim.models.Word2Vec.load(config['MODEL']['pre_train_model'])
    if len(all_doc) > 0:
        model_.min_count = int(config['MODEL']['min_count'])
        model_.build_vocab(all_doc, update=True)
        model_.train(all_doc, total_examples=model_.corpus_count, epochs=int(config['MODEL']['epoch']))

model_.save(config['BASIC']['embedding_dir'])
print('Time for learning the embedding model: %s seconds' % (time.time() - start_time))
print('Model saved. Done!')
