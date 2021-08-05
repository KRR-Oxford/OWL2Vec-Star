"""Console script for owl2vec_star."""
import configparser
import multiprocessing
import os
import random
import sys
import time

import click

import gensim
from owl2vec_star.lib.RDF2Vec_Embed import get_rdf2vec_walks
from owl2vec_star.lib.Label import pre_process_words, URI_parse
from owl2vec_star.lib.Onto_Projection import Reasoner, OntologyProjection

import nltk
nltk.download('punkt')


@click.group()
def main():
    pass


@main.command()
@click.option("--ontology_file", type=click.Path(exists=True), default=None, help="The input ontology for embedding")
@click.option("--embedding_dir", type=click.Path(exists=True), default=None, help="The output embedding directory")
@click.option("--config_file", type=click.Path(exists=True), default='default.cfg', help="Configuration file")
@click.option("--URI_Doc", help="Using URI document", is_flag=True)
@click.option("--Lit_Doc", help="Using literal document", is_flag=True)
@click.option("--Mix_Doc", help="Using mixture document", is_flag=True)
def standalone(ontology_file, embedding_dir, config_file, uri_doc, lit_doc, mix_doc):
    config = configparser.ConfigParser()
    config.read(click.format_filename(config_file))

    if ontology_file:
        config['BASIC']['ontology_file'] = click.format_filename(ontology_file)

    if embedding_dir:
        config['BASIC']['embedding_dir'] = click.format_filename(embedding_dir)

    if uri_doc:
        config['DOCUMENT']['URI_Doc'] = 'yes'
    if lit_doc:
        config['DOCUMENT']['Lit_Doc'] = 'yes'
    if mix_doc:
        config['DOCUMENT']['Mix_Doc'] = 'yes'
    if 'cache_dir' not in config['DOCUMENT']:
        config['DOCUMENT']['cache_dir'] = './cache'

    if not os.path.exists(config['DOCUMENT']['cache_dir']):
        os.mkdir(config['DOCUMENT']['cache_dir'])

    if 'embedding_dir' not in config['BASIC']:
        config['BASIC']['embedding_dir'] = os.path.join(config['DOCUMENT']['cache_dir'], 'output')

    start_time = time.time()
    if ('ontology_projection' in config['DOCUMENT'] and config['DOCUMENT']['ontology_projection'] == 'yes') or \
        'pre_entity_file' not in config['DOCUMENT'] or 'pre_axiom_file' not in config['DOCUMENT'] or \
        'pre_annotation_file' not in config['DOCUMENT']:
        print('\n Access the ontology ...')
        projection = OntologyProjection(config['BASIC']['ontology_file'], reasoner=Reasoner.STRUCTURAL,
                                        only_taxonomy=False,
                                        bidirectional_taxonomy=True, include_literals=True, avoid_properties=set(),
                                        additional_preferred_labels_annotations=set(),
                                        additional_synonyms_annotations=set(),
                                        memory_reasoner='13351')
    else:
        projection = None

    # Ontology projection
    if 'ontology_projection' in config['DOCUMENT'] and config['DOCUMENT']['ontology_projection'] == 'yes':
        print('\nCalculate the ontology projection ...')
        projection.extractProjection()
        onto_projection_file = os.path.join(config['DOCUMENT']['cache_dir'], 'projection.ttl')
        projection.saveProjectionGraph(onto_projection_file)
        ontology_file = onto_projection_file
    else:
        ontology_file = config['BASIC']['ontology_file']

    # Extract and save seed entities (classes and individuals)
    # Or read entities specified by the user
    if 'pre_entity_file' in config['DOCUMENT']:
        entities = [line.strip() for line in open(config['DOCUMENT']['pre_entity_file']).readlines()]
    else:
        print('\nExtract classes and individuals ...')
        projection.extractEntityURIs()
        classes = projection.getClassURIs()
        individuals = projection.getIndividualURIs()
        entities = classes.union(individuals)
        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'entities.txt'), 'w') as f:
            for e in entities:
                f.write('%s\n' % e)

    # Extract axioms in Manchester Syntax if it is not pre_axiom_file is not set
    if 'pre_axiom_file' not in config['DOCUMENT']:
        print('\nExtract axioms ...')
        projection.createManchesterSyntaxAxioms()
        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'axioms.txt'), 'w') as f:
            for ax in projection.axioms_manchester:
                f.write('%s\n' % ax)

    # If pre_annotation_file is set, directly read annotations
    # else, read annotations including rdfs:label and other literals from the ontology
    #   Extract annotations: 1) English label of each entity, by rdfs:label or skos:preferredLabel
    #                        2) None label annotations as sentences of the literal document
    uri_label, annotations = dict(), list()

    if 'pre_annotation_file' in config['DOCUMENT']:
        with open(config['DOCUMENT']['pre_annotation_file']) as f:
            for line in f.readlines():
                tmp = line.strip().split()
                if tmp[1] == 'http://www.w3.org/2000/01/rdf-schema#label':
                    uri_label[tmp[0]] = pre_process_words(tmp[2:])
                else:
                    annotations.append([tmp[0]] + tmp[2:])

    else:
        print('\nExtract annotations ...')
        projection.indexAnnotations()
        for e in entities:
            if e in projection.entityToPreferredLabels and len(projection.entityToPreferredLabels[e]) > 0:
                label = list(projection.entityToPreferredLabels[e])[0]
                uri_label[e] = pre_process_words(words=label.split())
        for e in entities:
            if e in projection.entityToAllLexicalLabels:
                for v in projection.entityToAllLexicalLabels[e]:
                    if (v is not None) and \
                        (not (e in projection.entityToPreferredLabels and v in projection.entityToPreferredLabels[e])):
                        annotation = [e] + v.split()
                        annotations.append(annotation)

        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'annotations.txt'), 'w') as f:
            for e in projection.entityToPreferredLabels:
                for v in projection.entityToPreferredLabels[e]:
                    f.write('%s preferred_label %s\n' % (e, v))
            for a in annotations:
                f.write('%s\n' % ' '.join(a))

    # read URI document
    # two parts: walks, axioms (if the axiom file exists)
    walk_sentences, axiom_sentences, URI_Doc = list(), list(), list()
    if 'URI_Doc' in config['DOCUMENT'] and config['DOCUMENT']['URI_Doc'] == 'yes':
        print('\nGenerate URI document ...')
        walks_ = get_rdf2vec_walks(onto_file=ontology_file, walker_type=config['DOCUMENT']['walker'],
                                   walk_depth=int(config['DOCUMENT']['walk_depth']), classes=entities)
        print('Extracted %d walks for %d seed entities' % (len(walks_), len(entities)))
        walk_sentences += [list(map(str, x)) for x in walks_]

        axiom_file = os.path.join(config['DOCUMENT']['cache_dir'], 'axioms.txt')
        if os.path.exists(axiom_file):
            for line in open(axiom_file).readlines():
                axiom_sentence = [item for item in line.strip().split()]
                axiom_sentences.append(axiom_sentence)
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
            return [item.lower()]

    # read literal document
    # two parts: literals in the annotations (subject's label + literal words)
    #            replacing walk/axiom sentences by words in their labels
    Lit_Doc = list()
    if 'Lit_Doc' in config['DOCUMENT'] and config['DOCUMENT']['Lit_Doc'] == 'yes':
        print('\nGenerate literal document ...')
        for annotation in annotations:
            processed_words = pre_process_words(annotation[1:])
            if len(processed_words) > 0:
                Lit_Doc.append(label_item(item=annotation[0]) + processed_words)
        print('Extracted %d annotation sentences' % len(Lit_Doc))

        for sentence in walk_sentences:
            lit_sentence = list()
            for item in sentence:
                lit_sentence += label_item(item=item)
            Lit_Doc.append(lit_sentence)

        for sentence in axiom_sentences:
            lit_sentence = list()
            for item in sentence:
                lit_sentence += label_item(item=item)
            Lit_Doc.append(lit_sentence)

    # read mixture document
    # for each axiom/walk sentence
    #   -    all): for each entity, keep its entity URI, replace the others by label words
    #   - random): randomly select one entity, keep its entity URI, replace the others by label words
    Mix_Doc = list()
    if 'Mix_Doc' in config['DOCUMENT'] and config['DOCUMENT']['Mix_Doc'] == 'yes':
        print('\nGenerate mixture document ...')
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

    print('URI_Doc: %d, Lit_Doc: %d, Mix_Doc: %d' % (len(URI_Doc), len(Lit_Doc), len(Mix_Doc)))
    all_doc = URI_Doc + Lit_Doc + Mix_Doc

    print('Time for document construction: %s seconds' % (time.time() - start_time))
    random.shuffle(all_doc)


    #Save all_doc (optional)
    #with open(os.path.join(config['DOCUMENT']['cache_dir'], 'document_sentences.txt'), 'w') as f:
    #    for sentence in all_doc:
    #        for w in sentence:
    #            f.write('%s ' % w)
    #        f.write('\n')
    #    f.close()


    # learn the language model (train a new model or fine tune the pre-trained model)
    start_time = time.time()
    if 'pre_train_model' not in config['MODEL'] or not os.path.exists(config['MODEL']['pre_train_model']):
        print('\nTrain the language model ...')
        model_ = gensim.models.Word2Vec(all_doc, size=int(config['MODEL']['embed_size']),
                                        window=int(config['MODEL']['window']),
                                        workers=multiprocessing.cpu_count(),
                                        sg=1, iter=int(config['MODEL']['iteration']),
                                        negative=int(config['MODEL']['negative']),
                                        min_count=int(config['MODEL']['min_count']), seed=int(config['MODEL']['seed']))
    else:
        print('\nFine-tune the pre-trained language model ...')
        model_ = gensim.models.Word2Vec.load(config['MODEL']['pre_train_model'])
        if len(all_doc) > 0:
            model_.min_count = int(config['MODEL']['min_count'])
            model_.build_vocab(all_doc, update=True)
            model_.train(all_doc, total_examples=model_.corpus_count, epochs=int(config['MODEL']['epoch']))

    #Gensim format
    model_.save(config['BASIC']['embedding_dir']+"ontology.embeddings")
    #Txt format
    model_.wv.save_word2vec_format(config['BASIC']['embedding_dir']+"ontology.embeddings.txt", binary=False)
    
    print('Time for learning the language model: %s seconds' % (time.time() - start_time))
    print('Model saved. Done!')

    return 0


@main.command()
@click.option("--ontology_dir", type=click.Path(exists=True), default=None, help="The directory of input ontologies for embedding")
@click.option("--embedding_dir", type=click.Path(exists=True), default=None, help="The output embedding directory")
@click.option("--config_file", type=click.Path(exists=True), default='default_multi.cfg', help="Configuration file")
@click.option("--URI_Doc", help="Using URI document", is_flag=True)
@click.option("--Lit_Doc", help="Using literal document", is_flag=True)
@click.option("--Mix_Doc", help="Using mixture document", is_flag=True)
def standalone_multi(ontology_dir, embedding_dir, config_file, uri_doc, lit_doc, mix_doc):
    # read and combine configurations
    # overwrite the parameters in the configuration file by the command parameters
    config = configparser.ConfigParser()
    config.read(click.format_filename(config_file))

    if ontology_dir:
        config['BASIC']['ontology_dir'] = click.format_filename(ontology_dir)

    if embedding_dir:
        config['BASIC']['embedding_dir'] = click.format_filename(embedding_dir)

    if uri_doc:
        config['DOCUMENT']['URI_Doc'] = 'yes'
    if lit_doc:
        config['DOCUMENT']['Lit_Doc'] = 'yes'
    if mix_doc:
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
                            (not (e in projection.entityToPreferredLabels and v in projection.entityToPreferredLabels[
                                e])):
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

    # learn the language model (train a new model or fine tune the pre-trained model)
    start_time = time.time()
    if 'pre_train_model' not in config['MODEL'] or not os.path.exists(config['MODEL']['pre_train_model']):
        print('\n\nTrain the language model')
        model_ = gensim.models.Word2Vec(all_doc, size=int(config['MODEL']['embed_size']),
                                        window=int(config['MODEL']['window']),
                                        workers=multiprocessing.cpu_count(),
                                        sg=1, iter=int(config['MODEL']['iteration']),
                                        negative=int(config['MODEL']['negative']),
                                        min_count=int(config['MODEL']['min_count']), seed=int(config['MODEL']['seed']))
    else:
        print('\n\nFine-tune the pre-trained language model')
        model_ = gensim.models.Word2Vec.load(config['MODEL']['pre_train_model'])
        if len(all_doc) > 0:
            model_.min_count = int(config['MODEL']['min_count'])
            model_.build_vocab(all_doc, update=True)
            model_.train(all_doc, total_examples=model_.corpus_count, epochs=int(config['MODEL']['epoch']))


    #Gensim format
    model_.save(config['BASIC']['embedding_dir']+"ontology.embeddings")
    #Txt format
    model_.wv.save_word2vec_format(config['BASIC']['embedding_dir']+"ontology.embeddings.txt", binary=False)
    
    print('Time for learning the language model: %s seconds' % (time.time() - start_time))
    print('Model saved. Done!')



if __name__ == "__main__":
    print("ciao")
    sys.exit(main())  # pragma: no cover
