import json
import logging
import os
import pickle
import re
import tempfile
import time

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, precision_recall_fscore_support

from nomothesia_nlp.common.text_vectorization.vectorizer import W2VVectorizer, ELMoVectorizer, BERTVectorizer
from nomothesia_nlp.data import MODELS_DIR, DATA_SET_DIR, DATA_DIR, LOGGING_DIR
from nomothesia_nlp.document_model.model import Token
from nomothesia_nlp.experiments.configurations.configuration import Configuration
from nomothesia_nlp.metrics.retrieval import mean_recall_k, mean_precision_k, mean_ndcg_score, mean_rprecision_k
from nomothesia_nlp.neural_networks.task_specific_networks.document_classification import DocumentClassification
from nomothesia_nlp.neural_networks.task_specific_networks.label_driven_classification import LabelDrivenClassification
from nomothesia_nlp.neural_networks.utilities import probas_to_classes
from ..experiment import Experiment
import glob
from collections import Counter
import tqdm

LOGGER = logging.getLogger(__name__)


class RaptarchisClassification(Experiment):

    def __init__(self):
        super().__init__()
        if 'elmo' in Configuration['model']['token_encoding']:
            self.vectorizer = ELMoVectorizer()
            self.vectorizer2 = W2VVectorizer(w2v_model=Configuration['model']['embeddings'])
        elif 'bert' in Configuration['model']['architecture'].lower():
            self.vectorizer = BERTVectorizer()
        else:
            self.vectorizer = W2VVectorizer(w2v_model=Configuration['model']['embeddings'])
        self.load_label_descriptions()

    def load_label_descriptions(self):
        LOGGER.info('Load labels\' data')
        LOGGER.info('-------------------')

        # Load train dataset and count labels
        train_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'train', '*.json'))
        train_counts = Counter()
        for filename in tqdm.tqdm(train_files):
            with open(filename) as file:
                data = json.load(file)
                train_counts[data[Configuration['task']['class_collection']]] += 1

        train_concepts = set(list(train_counts))

        frequent, few = [], []
        for i, (label, count) in enumerate(train_counts.items()):
            if count > Configuration['sampling']['few_threshold']:
                frequent.append(label)
            else:
                few.append(label)

        # Load dev/test datasets and count labels
        rest_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'dev', '*.json'))
        rest_files += glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'test', '*.json'))
        rest_concepts = set()
        for filename in tqdm.tqdm(rest_files):
            with open(filename) as file:
                data = json.load(file)
                rest_concepts.add(data[Configuration['task']['class_collection']])

        # Compute zero-shot group
        zero = list(rest_concepts.difference(train_concepts))

        self.label_ids = dict()
        self.margins = [(0, len(frequent) + len(few) + len(zero))]
        k = 0
        for group in [frequent, few, zero]:
            self.margins.append((k, k + len(group)))
            for concept in group:
                self.label_ids[concept] = k
                k += 1
        self.margins[-1] = (self.margins[-1][0], len(frequent) + len(few) + len(zero))

        # Load label descriptors
        with open(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], '{}_{}.json'.format(Configuration['task']['dataset'], Configuration['task']['task_language']))) as file:
            data = json.load(file)

        label_terms = []
        self.label_terms_text = []
        for i, (label, index) in enumerate(self.label_ids.items()):
                label_terms.append([Token(None, 0, 0, 'UNK', 'UNK', token)
                                    for token in label.split()])
                self.label_terms_text.append(label)
        self.label_terms_ids = self.vectorizer.vectorize_inputs(label_terms, max_sequence_size=Configuration['sampling']['max_label_size'],
                                                                features=['word'])
        LOGGER.info('Labels shape:    {}'.format(self.label_terms_ids.shape))
        LOGGER.info('Frequent labels: {}'.format(len(frequent)))
        LOGGER.info('Few labels:      {}'.format(len(few)))
        LOGGER.info('Zero labels:     {}'.format(len(zero)))

        # Compute label hierarchy depth and build labels' graph
        self.labels_graph = np.zeros((len(self.label_ids), len(self.label_ids)), dtype=np.float32)

    def process_dataset(self, documents):

        samples = []
        targets = []
        for document in documents:
            if Configuration['sampling']['hierarchical']:
                samples.append(document.sentences)
            else:
                samples.append(document.tokens)
            targets.append(document.tags)

        del documents
        return samples, targets

    def encode_dataset(self, sequences, tags, max_sequences_size=None, max_sequence_size=None):

        max_sequences_size = min(max_sequences_size, Configuration['sampling']['max_sequences_size'])
        max_sequence_size = min(max_sequence_size, Configuration['sampling']['max_sequence_size'])

        if Configuration['sampling']['hierarchical']:
            samples = np.zeros((len(sequences), max_sequences_size, max_sequence_size,), dtype=np.int32)
            if Configuration['model']['architecture'] in ['TRANSFORMERS', 'BERT']:
                samples_pos = np.repeat(np.arange(512, dtype=np.int32).reshape(1, -1), len(sequences), 0)

            targets = np.zeros((len(sequences), len(self.label_ids)), dtype=np.int32)
            for i, (sub_sequences, document_tags) in enumerate(zip(sequences, tags)):
                sample = self.vectorizer.vectorize_inputs(sub_sequences[:max_sequences_size],
                                                          max_sequence_size=max_sequence_size,
                                                          features=['word'])
                samples[i, :len(sample)] = sample
                for tag in document_tags:
                    if tag.name in self.label_ids:
                        targets[i][self.label_ids[tag.name]] = 1
            samples = np.asarray(samples)
        else:
            samples = self.vectorizer.vectorize_inputs(sequences,
                                                       max_sequence_size=max_sequence_size,
                                                       features=['word'])

            if 'elmo' in Configuration['model']['token_encoding']:
                samples2 = self.vectorizer2.vectorize_inputs(sequences,
                                                             max_sequence_size=max_sequence_size,
                                                             features=['word'])

            if Configuration['model']['architecture'] == 'TRANSFORMERS':
                 samples_pos = np.repeat(np.arange(samples.shape[1], dtype=np.int32).reshape(1, -1), samples.shape[0], 0)

            targets = np.zeros((len(sequences), len(self.label_ids)), dtype=np.int32)
            for i, (document_tags) in enumerate(tags):
                for tag in document_tags:
                    if tag.name in self.label_ids:
                        targets[i][self.label_ids[tag.name]] = 1

        del sequences, tags

        if Configuration['model']['architecture'] == 'TRANSFORMERS':
            return [samples, samples_pos], targets
        elif 'BERT' in Configuration['model']['architecture']:
            return samples, targets

        return samples, targets

    def run_operation(self):
        LOGGER.info('\n---------------- {} Starting ----------------'.format(Configuration['task']['operation_mode']))

        if Configuration['task']['operation_mode'] == 'train':
            LOGGER.info('\n---------------- Simple training ----------------')
            for param_name, value in Configuration['model'].items():
                LOGGER.info('\t{}: {}'.format(param_name, value))

            self.train()

        elif Configuration['task']['operation_mode'] == 'rule_based':
            LOGGER.info('\n---------------- Rule-based ----------------')
            for param_name, value in Configuration['model'].items():
                LOGGER.info('\t{}: {}'.format(param_name, value))

            self.rule_based()

        elif Configuration['task']['operation_mode'] == 'hyperopt':
            LOGGER.info('\n---------------- Hyper Optimization Parameters ----------------')
            LOGGER.info('=> Architecture: {}'.format(Configuration['model']['architecture']))
            LOGGER.info('=> Search space')
            for param_name, value in Configuration().search_space().items():
                LOGGER.info('\t{}: {}'.format(param_name, value))
            LOGGER.info('=> Score to track: {}'.format(Configuration['hyper_optimization']['score_to_track']))
            self.hyper_optimization()

        elif Configuration['task']['operation_mode'] == 'evaluate':
            # Load test data
            LOGGER.info('Load test data')
            LOGGER.info('------------------------------')

            test_documents = self.load_dataset('test_v3')
            limit = len(test_documents) % Configuration['model']['batch_size'] if Configuration['model']['architecture'] in ['TRANSFORMERS', 'BERT'] else 0
            test_samples, test_tags = self.process_dataset(test_documents if not limit else test_documents[:-limit])
            test_samples, test_targets = self.encode_dataset(test_samples, test_tags)

            self.evaluate(documents=test_documents, samples=test_samples, targets=test_targets)
        elif Configuration['task']['operation_mode'] == 'plot_attentions':
            # Load test data
            LOGGER.info('Load test data')
            LOGGER.info('------------------------------')

            test_documents = self.load_dataset('test_v3')
            limit = len(test_documents) % Configuration['model']['batch_size'] if Configuration['model']['architecture'] in ['TRANSFORMERS', 'BERT'] else 0
            test_samples, test_tags = self.process_dataset(test_documents if not limit else test_documents[:-limit])
            test_samples, test_targets = self.encode_dataset(test_samples, test_tags)

            self.plot_attentions(documents=test_documents, samples=test_samples, targets=test_tags)

        else:
            raise Exception('Operation mode "{}" is not supported'.format(Configuration['task']['operation_mode']))

    def train(self):
        # Load training/validation data
        LOGGER.info('Load training/validation data')
        LOGGER.info('------------------------------')

        documents = self.load_dataset('train')
        limit = len(documents) % Configuration['model']['batch_size'] if Configuration['model']['architecture'] in ['TRANSFORMERS', 'BERT'] else 0
        train_samples, train_tags = self.process_dataset(documents if not limit else documents[:-limit])
        train_generator = SampleGenerator(train_samples, train_tags, experiment=self, batch_size=Configuration['model']['batch_size'])

        documents = self.load_dataset('dev')
        limit = len(documents) % Configuration['model']['batch_size'] if Configuration['model']['architecture'] in ['TRANSFORMERS', 'BERT'] else 0
        val_samples, val_tags = self.process_dataset(documents if not limit else documents[:-limit])
        val_generator = SampleGenerator(val_samples, val_tags, experiment=self, batch_size=Configuration['model']['batch_size'])

        # Compile neural network
        LOGGER.info('Compile neural network')
        LOGGER.info('------------------------------')
        if 'label' in Configuration['model']['architecture'].lower():
            network = LabelDrivenClassification(self.label_terms_ids)
        else:
            network = DocumentClassification(self.label_terms_ids)

        network.compile(n_hidden_layers=Configuration['model']['n_hidden_layers'],
                        hidden_units_size=Configuration['model']['hidden_units_size'],
                        dropout_rate=Configuration['model']['dropout_rate'],
                        word_dropout_rate=Configuration['model']['word_dropout_rate'],
                        lr=Configuration['model']['lr'])

        network.summary(line_length=200, print_fn=LOGGER.info)

        with tempfile.NamedTemporaryFile(delete=True) as w_fd:
            weights_file = w_fd.name

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(filepath=weights_file, monitor='val_loss', mode='auto',
                                           verbose=1, save_best_only=True, save_weights_only=True)

        # Fit model
        LOGGER.info('Fit model')
        LOGGER.info('-----------')
        start_time = time.time()
        fit_history = network.fit(train_generator,
                                            validation_data=val_generator,
                                            workers=os.cpu_count(),
                                            epochs=Configuration['model']['epochs'],
                                            callbacks=[early_stopping, model_checkpoint])

        # Save model
        model_name = '{}_{}_{}'.format(
                    Configuration['task']['dataset'].upper(), 'HIERARCHICAL' if Configuration['sampling']['hierarchical'] else 'FLAT',
                    Configuration['model']['architecture'].upper())
        # network.dump(model_name=model_name)

        best_epoch = np.argmin(fit_history.history['val_loss']) + 1
        n_epochs = len(fit_history.history['val_loss'])
        val_loss_per_epoch = '- ' + ' '.join('-' if fit_history.history['val_loss'][i] < np.min(fit_history.history['val_loss'][:i])
                                             else '+' for i in range(1, len(fit_history.history['val_loss'])))
        LOGGER.info('\nBest epoch: {}/{}'.format(best_epoch, n_epochs))
        LOGGER.info('Val loss per epoch: {}\n'.format(val_loss_per_epoch))

        del train_generator

        LOGGER.info('Load valid data')
        LOGGER.info('------------------------------')
        val_samples, val_targets = self.encode_dataset(val_samples, val_tags,
                                                       max_sequences_size=Configuration['sampling']['max_sequences_size'],
                                                       max_sequence_size=Configuration['sampling']['max_sequence_size'])
        self.calculate_performance(network=network, true_samples=val_samples, true_targets=val_targets)
        del val_samples, val_targets, val_generator

        LOGGER.info('Load test data')
        LOGGER.info('------------------------------')

        test_documents = self.load_dataset('test')
        limit = len(test_documents) % Configuration['model']['batch_size'] if Configuration['model']['architecture'] in ['TRANSFORMERS', 'BERT'] else 0
        test_samples, test_tags = self.process_dataset(test_documents if not limit else test_documents[:-limit])
        test_samples, test_targets = self.encode_dataset(test_samples, test_tags,
                                                         max_sequences_size=Configuration['sampling']['max_sequences_size'],
                                                         max_sequence_size=Configuration['sampling']['max_sequence_size'])
        self.calculate_performance(network=network, true_samples=test_samples, true_targets=test_targets, name=model_name)

        total_time = time.time() - start_time
        LOGGER.info('\nTotal Training Time: {} secs'.format(total_time))

    def hyper_optimization(self):
        self._trial_no = 0
        # Load training/validation data
        LOGGER.info('Load training/validation/test data')
        LOGGER.info('------------------------------')

        documents = self.load_dataset('train')
        self.split_limit = len(documents)
        documents.extend(self.load_dataset('dev'))

        test_documents = self.load_dataset('test')

        if 'label' in Configuration['model']['architecture'].lower():
            network = LabelDrivenClassification(self.label_terms_ids)
        else:
            network = DocumentClassification(self.label_terms_ids)

        search_space = dict([(key, hp.choice(key, value)) for key, value in Configuration().search_space().items()])
        space_item = pyll.rec_eval({key: value.pos_args[-1] for key, value in search_space.items()})

        network.compile(n_hidden_layers=space_item['n_hidden_layers'],
                        hidden_units_size=space_item['hidden_units_size'],
                        dropout_rate=space_item['dropout_rate'],
                        word_dropout_rate=space_item['word_dropout_rate'],
                        lr=space_item['learning_rate'])

        network.summary(line_length=200, print_fn=LOGGER.info)

        # Start hyper-opt trials
        while True:
            try:
                trials = pickle.load(open(Configuration['hyper_optimization']['log_name'], 'rb'))
                max_evals = len(trials.trials) + 1
            except FileNotFoundError:
                trials = Trials()
                max_evals = 1
            self._trial_no = max_evals
            if max_evals > Configuration['hyper_optimization']['trials']:
                break
            fmin(fn=lambda space_item: self.optimization_function(network=network,  documents=documents,
                                                                  test_documents=test_documents, current_space=space_item),
                 space=search_space,
                 algo=tpe.suggest,
                 max_evals=max_evals,
                 trials=trials)
            with open(Configuration['hyper_optimization']['log_name'], 'wb') as f:
                pickle.dump(trials, f)

        LOGGER.info('\n--------------------- Results Summary ------------------')
        for t in trials.results:
            conf = t['results']['configuration']
            average_statistics = t['results']['average_statistics']

            log_msg = 'Trial {:>2}/{} HL={:1}  HU={:3}  BS={:<3}  D={:<3}  WD={:<4}  AM={}  LR={:<5}'.format(
                t['trial_no'], Configuration['hyper_optimization']['trials'], str(conf['n_hidden_layers']),
                str(conf['hidden_units_size']), conf['batch_size'], conf['dropout_rate'], conf['word_dropout_rate'],
                conf['attention_mechanism'], conf['learning_rate'])
            log_msg += '\tFolds Average:  '
            log_msg += 'Val:  Overall Micro  {}  Overall {}   '.format(
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['validation']['Overall']['micro'][metric]['mean'])
                         for metric in ['P', 'R', 'F1']]),
                ''.join(['RP@{}={:.3f}  '.format(i, average_statistics['validation']['Overall']['RP@'][i]['mean'])
                         for i in range(1, Configuration['sampling']['evaluation@k'] + 1)]))
            log_msg += 'Test:  Overall Micro  {}  Overall {}'.format(
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['test']['Overall']['micro'][metric]['mean']) for metric in ['P', 'R', 'F1']]),
                ''.join(['RP@{}={:.3f}  '.format(i, average_statistics['test']['Overall']['RP@'][i]['mean'])
                         for i in range(1, Configuration['sampling']['evaluation@k'] + 1)]))
            LOGGER.info(log_msg)

        LOGGER.info('\n\n--------------------- Results Summary Best to Worst ------------------')
        for t in sorted(trials.results, key=lambda trial: trial['loss'], reverse=False):
            conf = t['results']['configuration']
            average_statistics = t['results']['average_statistics']

            log_msg = 'Trial {:>2}/{} HL={:1}  HU={:3}  BS={:<3}  D={:<3}  WD={:<4}  AM={}  LR={:<5}'.format(
                t['trial_no'], Configuration['hyper_optimization']['trials'], str(conf['n_hidden_layers']),
                str(conf['hidden_units_size']), conf['batch_size'], conf['dropout_rate'], conf['word_dropout_rate'],
                conf['attention_mechanism'], conf['learning_rate'])
            log_msg += '\tFolds Average:  '
            log_msg += 'Val:  Overall Micro  {}  Overall {}   '.format(
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['validation']['Overall']['micro'][metric]['mean'])
                         for metric in ['P', 'R', 'F1']]),
                ''.join(['RP@{}={:.3f}  '.format(i, average_statistics['validation']['Overall']['RP@'][i]['mean'])
                         for i in range(1, Configuration['sampling']['evaluation@k'] + 1)]))
            log_msg += 'Test:  Overall Micro  {}  Overall {}'.format(
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['test']['Overall']['micro'][metric]['mean']) for metric in ['P', 'R', 'F1']]),
                ''.join(['RP@{}={:.3f}  '.format(i, average_statistics['test']['Overall']['RP@'][i]['mean'])
                         for i in range(1, Configuration['sampling']['evaluation@k'] + 1)]))
            LOGGER.info(log_msg)

        trials_training_time = sum([trial['results']['time'] for trial in trials.results])
        LOGGER.info('\nHyperopt search took {} days {}\n\n'.format(int(trials_training_time / (24 * 60 * 60)),
                                                                   time.strftime("%H:%M:%S", time.gmtime(trials_training_time))))

    def optimization_function(self, network,  documents, test_documents, current_space):
        trial_start = time.time()
        LOGGER.info(
            '\n' + '=' * 110 + '\nTrial {:>2}/{}:  HL={:1}  HU={:3}  BS={:<3}  D={:<3}  WD={:<4}  AM={}  LR={:<5}'.format(
                self._trial_no, Configuration['hyper_optimization']['trials'], str(current_space['n_hidden_layers']),
                str(current_space['hidden_units_size']),
                current_space['batch_size'], current_space['dropout_rate'], current_space['word_dropout_rate'],
                current_space['attention_mechanism'], current_space['learning_rate']) + '\n' + '=' * 110)

        # Create folder to save all trial models
        if not os.path.exists(os.path.join(MODELS_DIR, 'hyperopt', Configuration['task']['log_name'])):
            os.makedirs(os.path.join(MODELS_DIR, 'hyperopt', Configuration['task']['log_name']))

        score_to_track = Configuration['hyper_optimization']['score_to_track']

        if Configuration['hyper_optimization']['folds'] != 1:
            self.split_limit = int(len(documents) * (1 - Configuration['sampling']['validation_size']))

        # Initialize the structure that will hold the optimization statistics
        statistics = {}
        for dataset in ['validation', 'test']:
            statistics[dataset] = {}
            for freq in ['Overall', 'Frequent', 'Few', 'Zero']:
                statistics[dataset][freq] = {}
                for average_type in ['micro', 'macro', 'weighted']:
                    statistics[dataset][freq][average_type] = {}
                    for metric in ['P', 'R', 'F1']:
                        statistics[dataset][freq][average_type][metric] = []

                for metric in ['R@', 'P@', 'RP@', 'NDCG@']:
                    statistics[dataset][freq][metric] = {}
                    for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                        statistics[dataset][freq][metric][i] = []

        fold_loss = []
        limit = len(test_documents) % Configuration['model']['batch_size'] if Configuration['model']['architecture'] in ['TRANSFORMERS', 'BERT'] else 0
        test_samples, test_tags = self.process_dataset(test_documents if not limit else test_documents[:-limit])
        test_samples, test_targets = self.encode_dataset(test_samples, test_tags,
                                                         max_sequences_size=Configuration['sampling']['max_sequences_size'],
                                                         max_sequence_size=Configuration['sampling']['max_sequence_size'])

        # Train the model with  the same configuration for N folds
        for fold_no in range(Configuration['hyper_optimization']['folds']):
            LOGGER.info('\n----- Fold: {0}/{1} -----\n'.format(fold_no + 1, Configuration['hyper_optimization']['folds']))

            indices = np.arange(len(documents))
            if Configuration['hyper_optimization']['folds'] != 1:
                np.random.seed(fold_no)
                np.random.shuffle(indices)

            limit = len(documents) % Configuration['model']['batch_size'] if Configuration['model']['architecture'] in ['TRANSFORMERS', 'BERT'] else 0
            if limit:
                documents = documents[:-limit]
            train_samples, train_tags = self.process_dataset(np.asarray(documents)[indices][:self.split_limit])
            train_generator = SampleGenerator(train_samples, train_tags, experiment=self, batch_size=current_space['batch_size'])

            val_samples, val_tags = self.process_dataset(np.asarray(documents)[indices][self.split_limit:])
            val_generator = SampleGenerator(val_samples, val_tags, experiment=self, batch_size=current_space['batch_size'])

            # Compile neural network
            network.compile(n_hidden_layers=current_space['n_hidden_layers'],
                            hidden_units_size=current_space['hidden_units_size'],
                            dropout_rate=current_space['dropout_rate'],
                            word_dropout_rate=current_space['word_dropout_rate'],
                            lr=current_space['learning_rate'])

            # Add callbacks (early stopping, model checkpoint)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            with tempfile.NamedTemporaryFile(delete=True) as w_fd:
                weights_file = w_fd.name

                model_checkpoint = ModelCheckpoint(filepath=weights_file, monitor='val_loss', mode='auto',
                                                   verbose=1, save_best_only=True, save_weights_only=True)

                fit_history = network.fit(train_generator,
                                                    validation_data=val_generator,
                                                    workers=os.cpu_count(),
                                                    epochs=Configuration['hyper_optimization']['epochs'],
                                                    callbacks=[early_stopping, model_checkpoint])

            # Save model
            # network.dump(folder=os.path.join('hyperopt', Configuration['task']['log_name']),
            #              model_name='{}_{}_{}_trial_{}_fold_{}'.format(
            #                  Configuration['task']['dataset'].upper(), 'HIERARCHICAL' if Configuration['sampling']['hierarchical'] else 'FLAT',
            #                  Configuration['model']['architecture'].upper(), self._trial_no, fold_no + 1))

            best_epoch = np.argmin(fit_history.history['val_loss']) + 1
            n_epochs = len(fit_history.history['val_loss'])
            val_loss_per_epoch = '- ' + ' '.join('-' if fit_history.history['val_loss'][i] < np.min(fit_history.history['val_loss'][:i])
                                                 else '+' for i in range(1, len(fit_history.history['val_loss'])))
            LOGGER.info('\nBest epoch: {}/{}'.format(best_epoch, n_epochs))
            LOGGER.info('Val loss per epoch: {}\n'.format(val_loss_per_epoch))

            # Calculate validation performance
            LOGGER.info('\n----- Validation Classification Results -----')
            val_samples, val_targets = self.encode_dataset(val_samples, val_tags,
                                                         max_sequences_size=Configuration['sampling']['max_sequences_size'],
                                                         max_sequence_size=Configuration['sampling']['max_sequence_size'])
            val_report_statistics = self.calculate_performance(network=network, true_samples=val_samples, true_targets=val_targets)

            for freq in ['Overall', 'Frequent', 'Few', 'Zero']:
                for average_type in ['micro', 'macro', 'weighted']:
                    for metric in ['P', 'R', 'F1']:
                        statistics['validation'][freq][average_type][metric].append(val_report_statistics[freq][average_type][metric])

                for metric in ['R@', 'P@', 'RP@', 'NDCG@']:
                    for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                        statistics['validation'][freq][metric][i].append(val_report_statistics[freq][metric][i])

            # Calculate test performance
            LOGGER.info('\n----- Test Classification Results -----')
            test_report_statistics = self.calculate_performance(network=network, true_samples=test_samples, true_targets=test_targets, name= self._trial_no)

            for freq in ['Overall', 'Frequent', 'Few', 'Zero']:
                for average_type in ['micro', 'macro', 'weighted']:
                    for metric in ['P', 'R', 'F1']:
                        statistics['test'][freq][average_type][metric].append(test_report_statistics[freq][average_type][metric])

                for metric in ['R@', 'P@', 'RP@', 'NDCG@']:
                    for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                        statistics['test'][freq][metric][i].append(test_report_statistics[freq][metric][i])

            if 'micro' in score_to_track or 'macro' in score_to_track or 'weighted' in score_to_track:
                average_type, metric = score_to_track.split('-')
                fold_loss.append(1 - val_report_statistics['Overall'][average_type][metric])
            elif '@' in score_to_track:
                metric, k = score_to_track.split('@')
                fold_loss.append(1 - val_report_statistics['Overall']['{}@'.format(metric)][int(k)])
            else:
                raise Exception('Score to track: {} is not supported'.format(score_to_track))

        average_statistics = {}
        for dataset in ['validation', 'test']:
            average_statistics[dataset] = {}
            for freq in ['Overall', 'Frequent', 'Few', 'Zero']:
                average_statistics[dataset][freq] = {}
                for average_type in ['micro', 'macro', 'weighted']:
                    average_statistics[dataset][freq][average_type] = {}
                    for metric in ['P', 'R', 'F1']:
                        average_statistics[dataset][freq][average_type][metric] = \
                            self.calculate_std_standard_error(statistics[dataset][freq][average_type][metric])

                for metric in ['R@', 'P@', 'RP@', 'NDCG@']:
                    average_statistics[dataset][freq][metric] = {}
                    for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                        average_statistics[dataset][freq][metric][i] = self.calculate_std_standard_error(statistics[dataset][freq][metric][i])

        if Configuration['hyper_optimization']['folds'] != 1:
            self.average_report(average_statistics)

        LOGGER.info('Trial training took {0} sec\n'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - trial_start))))

        current_space['trial_no'] = self._trial_no
        return {'loss': np.average(fold_loss),
                'status': STATUS_OK,
                'trial_no': self._trial_no,
                'results': {'configuration': current_space, 'time': time.time() - trial_start,
                            'statistics': statistics, 'average_statistics': average_statistics}
                }

    def calculate_performance(self, network, true_samples, true_targets, name= None):

        if Configuration['model']['return_attention']:
            predictions = network.predict(true_samples,
                                          batch_size=Configuration['model']['batch_size']
                                          if Configuration['model']['architecture'] in ['TRANSFORMERS', 'BERT']
                                             or Configuration['model']['token_encoding'] == 'elmo' else None)[0]
        else:
            predictions = network.predict(true_samples,
                                          batch_size=Configuration['model']['batch_size']
                                          if Configuration['model']['architecture'] in ['TRANSFORMERS', 'BERT']
                                             or Configuration['model']['token_encoding'] == 'elmo' else None)

        if name:
            with open(os.path.join('{}.predictions'.format(name)), 'wb') as file:
                pickle.dump(predictions, file)

        pred_targets = probas_to_classes(predictions)

        if Configuration['task']['operation_mode'] != 'hyperopt':
            with open(os.path.join(LOGGING_DIR, Configuration['task']['operation_mode'].lower(),
                                   Configuration['task']['log_name'] + '_classification_report.txt'), 'w') as file:
                report = classification_report(y_true=true_targets, y_pred=pred_targets, digits=4)
                p, r, f1, _ = precision_recall_fscore_support(y_true=true_targets, y_pred=pred_targets, average='micro')
                report += 'avg / micro     {:.4f}    {:.4f}    {:.4f}\n'.format(p, r, f1)
                p, r, f1, _ = precision_recall_fscore_support(y_true=true_targets, y_pred=pred_targets, average='macro')
                report += 'avg / macro     {:.4f}    {:.4f}    {:.4f}\n'.format(p, r, f1)
                p, r, f1, _ = precision_recall_fscore_support(y_true=true_targets, y_pred=pred_targets, average='weighted')
                report += 'weighted        {:.4f}    {:.4f}    {:.4f}\n'.format(p, r, f1)
                file.write(report)

        report_statistics = {}
        for freq in ['Overall', 'Frequent', 'Few', 'Zero']:
            report_statistics[freq] = {}
            for average_type in ['micro', 'macro', 'weighted']:
                report_statistics[freq][average_type] = {}
                for metric in ['P', 'R', 'F1']:
                    report_statistics[freq][average_type][metric] = 0

            for metric in ['R@', 'P@', 'RP@', 'NDCG@']:
                report_statistics[freq][metric] = {}
                for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                    report_statistics[freq][metric][i] = 0

        template = 'R@{} : {:1.3f}   P@{} : {:1.3f}   RP@{} : {:1.3f}   NDCG@{} : {:1.3f}'

        # Overall
        for labels_range, frequency, message in zip(self.margins, ['Overall', 'Frequent', 'Few', 'Zero'],
                                                    ['Overall', 'Frequent Labels (>=50 Occurrences in train set)',
                                                     'Few-shot (<=50 Occurrences in train set)', 'Zero-shot (No Occurrences in train set)']):
            start, end = labels_range
            if start == end:
                continue
            LOGGER.info(message)
            LOGGER.info('----------------------------------------------------')
            for average_type in ['micro', 'macro', 'weighted']:
                p = report_statistics[frequency][average_type]['P'] = precision_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                r = report_statistics[frequency][average_type]['R'] = recall_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                f1 = report_statistics[frequency][average_type]['F1'] = f1_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                LOGGER.info('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}'.format(average_type, p, r, f1))

            for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                r_k = report_statistics[frequency]['R@'][i] = mean_recall_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                p_k = report_statistics[frequency]['P@'][i] = mean_precision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                rp_k = report_statistics[frequency]['RP@'][i] = mean_rprecision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                ndcg_k = report_statistics[frequency]['NDCG@'][i] = mean_ndcg_score(true_targets[:, start:end], predictions[:, start:end], k=i)
                LOGGER.info(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
            LOGGER.info('----------------------------------------------------')

        return report_statistics

    def average_report(self, average_statistics):
        LOGGER.info('\n---------- Average Classification Results ----------\n')

        template = 'R@{} : {}   P@{} : {}   RP@{} : {}   NDCG@{} : {}'

        for dataset in ['validation', 'test']:
            LOGGER.info('\n{} Classification Report\n'.format(dataset.capitalize()))
            for frequency in ['Overall', 'Frequent', 'Few', 'Zero']:
                LOGGER.info(frequency)
                LOGGER.info('----------------------------------------------------')
                for average_type in ['micro', 'macro', 'weighted']:
                    LOGGER.info('{:8} - Precision: {}   Recall: {}   F1: {}'.format(
                        average_type,
                        self.metric_to_str(average_statistics[dataset][frequency][average_type]['P']),
                        self.metric_to_str(average_statistics[dataset][frequency][average_type]['R']),
                        self.metric_to_str(average_statistics[dataset][frequency][average_type]['F1'])))

                for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                    LOGGER.info(template.format(i, self.metric_to_str(average_statistics[dataset][frequency]['R@'][i]),
                                                i, self.metric_to_str(average_statistics[dataset][frequency]['P@'][i]),
                                                i, self.metric_to_str(average_statistics[dataset][frequency]['RP@'][i]),
                                                i, self.metric_to_str(average_statistics[dataset][frequency]['NDCG@'][i])))
                LOGGER.info('----------------------------------------------------')

    def metric_to_str(self, metric):
        return '{:.3f} (std={:.4f} se={:.4f} rel_se={:.4f})'.format(metric['mean'], metric['std'], metric['st_error'], metric['rel_st_error'])

    def calculate_std_standard_error(self, values):
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        standard_error = std / np.sqrt(len(values))
        relative_standard_error = standard_error / (mean + np.finfo(float).eps)

        return {'mean': mean, 'std': std, 'st_error': standard_error, 'rel_st_error': relative_standard_error}

    def evaluate(self, documents, samples, targets):
        # Compile neural network
        LOGGER.info('Compile neural network')
        LOGGER.info('------------------------------')

        network = DocumentClassification(label_terms_ids=self.label_terms_ids)

        # network.compile(n_hidden_layers=Configuration['model']['n_hidden_layers'],
        #                 hidden_units_size=Configuration['model']['hidden_units_size'],
        #                 dropout_rate=Configuration['model']['dropout_rate'],
        #                 word_dropout_rate=Configuration['model']['word_dropout_rate'],
        #                 lr=Configuration['model']['lr'])

        network.load('/home/ichalkidis/nomothesia.nlp.training/nomothesia_nlp/data/models/train/EUROVOC_EN_FLAT_BERT.h5')

        network.summary(line_length=200, print_fn=LOGGER.info)

        self.calculate_performance(network=network, true_samples=samples, true_targets=targets, name='GRUS')

    def plot_attentions(self, documents, samples, targets):
        import tqdm
        with open(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'],
                               'eurovoc_{}.json'.format(Configuration['task']['task_language']))) as file:
            concepts = json.load(file)

        keys = list(self.label_ids.keys())
        neural_network = DocumentClassification(label_terms_ids=self.label_terms_ids)
        neural_network.load('/home/ichalkidis/nomothesia.nlp.training/nomothesia_nlp/data/models/train/EUROVOC_EN_FLAT_BI-GRUS_trial_1_fold_1.h5')
        neural_network.save_weights('/home/ichalkidis/WEIGHTS2.hdf5')

        neural_network.compile(n_hidden_layers=Configuration['model']['n_hidden_layers'],
                               hidden_units_size=Configuration['model']['hidden_units_size'],
                               dropout_rate=Configuration['model']['dropout_rate'],
                               word_dropout_rate=Configuration['model']['word_dropout_rate'],
                               lr=Configuration['model']['lr'])
        neural_network.load_weights('/home/ichalkidis/WEIGHTS2.hdf5')

        for i, sample in tqdm.tqdm(enumerate(samples)):
            if len(targets[i]) > 5 or len(documents[i].tokens) > 500:
                continue
            true = []
            for tag in targets[i]:
                if tag.name in keys:
                    true.append(tag.name)
            if Configuration['sampling']['hierarchical']:
                predictions, word_attention_scores, section_attention_scores = neural_network.predict(np.asarray([sample]))
            else:
                predictions, attention_scores = neural_network.predict(np.asarray([sample]))

            predictions = np.squeeze(predictions, axis=0)

            sample = documents[i]

            order = np.argsort(predictions)[::-1][:5]

            labels = []
            for el in order:
                labels.append(keys[el])

            if len(set(true).difference(set(labels))) > 1:
                continue

            html_file = open(os.path.join(DATA_DIR, 'logs', 'attentions', 'html', 'sample_{}.html'.format(i)), 'w')

            html_file.write('<p>Predicted Concepts: ')
            for label in labels:
                html_file.write('<b>{}</b> | '.format(concepts[label]['label']))
            html_file.write('</p>')
            html_file.write('<br/>')

            html_file.write('<p>True Concepts: ')
            for label in true:
                html_file.write('<b>{}</b> | '.format(concepts[label]['label']))
            html_file.write('</p>')
            html_file.write('<br/>')

            html_file.write('<!DOCTYPE html><html><head>'
                            '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">'
                            '<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>'
                            '<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>'
                            '<style type="text/css">body { padding: 200px}span { border: 0px solid;}</style> </head>' \
                            '<body>')
            if Configuration['sampling']['hierarchical']:
                word_attention_scores = np.squeeze(word_attention_scores, axis=0)
                section_attention_scores = np.squeeze(np.squeeze(section_attention_scores, axis=0), axis=1)
                html_file.write('<table>')
                mean_section = np.mean(section_attention_scores[:len(sample)])
                std_section = np.std(section_attention_scores[:len(sample)])
                for i, (sentence, section_attention_scores, section_attention_score) in enumerate(zip(sample, word_attention_scores, section_attention_scores)):
                    html_file.write('<tr><td style="min-width:20px;"></td><td><b>Section #{}</b></td><tr>'.format(i+1))
                    html_file.write('<tr>')
                    # Compute z-scores and print heat-map for section
                    if section_attention_score < mean_section:
                        html_file.write('<td style="min-width:20px;">')
                    else:
                        color_opacity = (section_attention_score - mean_section) / (3 * std_section)
                        html_file.write('<td style= "min-width:20px; background-color:rgba(255, 0, 0, {0:.1f});">  </span>'.format(color_opacity))
                    html_file.write('</td>')
                    html_file.write('<td>')
                    # Compute mean and standard deviation
                    mean = np.mean(section_attention_scores[:len(sentence)])
                    std = np.std(section_attention_scores[:len(sentence)])
                    # Compute z-scores and print heat-map for words
                    for attention_score, word in zip(section_attention_scores, sentence):
                        if word.token_text == '\n':
                            html_file.write('<br/>')
                        else:
                            if attention_score < mean:
                                html_file.write('<span>{} </span>'.format(word.token_text))
                            else:
                                color_opacity = (attention_score - mean) / (3 * std)
                                html_file.write('<span style= "background-color:rgba(255, 0, 0, {0:.1f});">{1} </span>'.format(color_opacity,
                                                                                                                               word.token_text))
                    html_file.write('</td>')
                    html_file.write('</tr>')
                html_file.write('</table>')
            elif 'LABEL' in Configuration['model']['architecture']:
                # Compute mean and standard deviation
                attention_scores = np.squeeze(attention_scores, axis=(0))
                means = np.zeros((attention_scores.shape[0],), dtype=np.float32)
                stds = np.zeros((attention_scores.shape[0],), dtype=np.float32)
                for k in range(len(attention_scores)):
                    means[k] = np.mean(attention_scores[i][:len(sample.tokens)])
                    stds[k] = np.std(attention_scores[i][:len(sample.tokens)])
                # Compute z-scores and print heat-map for words
                for k, word in enumerate(sample.tokens):
                    if word.token_text == '\n':
                        html_file.write('<br/>')
                    else:
                        colours = ['255, 0, 0', '66, 134, 244', '57, 219, 146', '244, 188, 66']
                        max = 0
                        top_index = 0
                        colour_id = 0
                        for j, index in enumerate(order[:4]):
                            if attention_scores[index][k] > max:
                                top_index = index
                                colour_id = j
                                max = attention_scores[index][k]
                        if attention_scores[top_index][k] < means[top_index]:
                            html_file.write('<span>{} </span>'.format(word.token_text))
                        else:
                            color_opacity = (attention_scores[top_index][k] - means[top_index]) / (3 * stds[top_index])
                            html_file.write('<span style= "background-color:rgba({0}, {1:.1f});">{2} </span>'.format(colours[colour_id], color_opacity,
                                                                                                                    word.token_text))
            else:
                # Compute mean and standard deviation
                attention_scores = np.squeeze(attention_scores, axis=(0, 2))
                mean = np.mean(attention_scores[:len(sample.tokens)])
                std = np.std(attention_scores[:len(sample.tokens)])
                # Compute z-scores and print heat-map for words
                for attention_score, word in zip(attention_scores, sample.tokens):
                    if word.token_text == '\n':
                        html_file.write('<br/>')
                    else:
                        if attention_score < mean:
                            html_file.write('<span>{} </span>'.format(word.token_text))
                        else:
                            color_opacity = (attention_score - mean) / (3 * std)
                            html_file.write('<span style= "background-color:rgba(255, 0, 0, {0:.1f});">{1} </span>'.format(color_opacity,
                                                                                                                           word.token_text))
            html_file.write('</body>')
            html_file.close()

    def rule_based(self):
        import tqdm
        # Load training/validation data
        LOGGER.info('Load validation data')
        LOGGER.info('------------------------------')

        documents = self.load_dataset('test_v3')
        val_targets = np.zeros((len(documents), len(self.label_ids)), dtype=np.int32)
        for i, document in enumerate(documents):
            for j, tag in enumerate(document.tags):
                if tag.name in self.label_ids:
                    val_targets[i][self.label_ids[tag.name]] = 1

        val_pred_targets = np.zeros((len(documents), len(self.label_ids)), dtype=np.float32)
        for i, document in tqdm.tqdm(enumerate(documents)):
            for j, label_text in enumerate(self.label_terms_text):
                pattern = re.sub(' +', '\s+', label_text.strip()).lower()
                if re.search(pattern, document.text.lower()):
                    val_pred_targets[i][j] = 1.0

        report_statistics = {}
        for freq in ['Overall', 'Frequent', 'Few', 'Zero']:
            report_statistics[freq] = {}
            for average_type in ['micro', 'macro', 'weighted']:
                report_statistics[freq][average_type] = {}
                for metric in ['P', 'R', 'F1']:
                    report_statistics[freq][average_type][metric] = 0

            for metric in ['R@', 'P@', 'RP@', 'NDCG@']:
                report_statistics[freq][metric] = {}
                for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                    report_statistics[freq][metric][i] = 0

        template = 'R@{} : {:1.3f}   P@{} : {:1.3f}   RP@{} : {:1.3f}   NDCG@{} : {:1.3f}'

        # Overall
        for labels_range, frequency, message in zip([(0, 4271), (0, 746), (746, 4107), (4107, 4271)], ['Overall', 'Frequent', 'Few', 'Zero'],
                                                    ['Overall', 'Frequent Labels (>=50 Occurrences in train set)',
                                                     'Few-shot (<=50 Occurrences in train set)', 'Zero-shot (No Occurrences in train set)']):
            start, end = labels_range
            LOGGER.info(message)
            LOGGER.info('----------------------------------------------------')
            for average_type in ['micro', 'macro', 'weighted']:
                p = report_statistics[frequency][average_type]['P'] = precision_score(val_targets[:, start:end], val_pred_targets[:, start:end],
                                                                                      average=average_type)
                r = report_statistics[frequency][average_type]['R'] = recall_score(val_targets[:, start:end], val_pred_targets[:, start:end],
                                                                                   average=average_type)
                f1 = report_statistics[frequency][average_type]['F1'] = f1_score(val_targets[:, start:end], val_pred_targets[:, start:end],
                                                                                 average=average_type)
                LOGGER.info('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}'.format(average_type, p, r, f1))

            for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                r_k = report_statistics[frequency]['R@'][i] = mean_recall_k(val_targets[:, start:end], val_pred_targets[:, start:end], k=i)
                p_k = report_statistics[frequency]['P@'][i] = mean_precision_k(val_targets[:, start:end], val_pred_targets[:, start:end], k=i)
                rp_k = report_statistics[frequency]['RP@'][i] = mean_rprecision_k(val_targets[:, start:end], val_pred_targets[:, start:end], k=i)
                ndcg_k = report_statistics[frequency]['NDCG@'][i] = mean_ndcg_score(val_targets[:, start:end], val_pred_targets[:, start:end], k=i)
                LOGGER.info(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
            LOGGER.info('----------------------------------------------------')


class SampleGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, samples, targets, experiment, batch_size=32, shuffle=True):
        """Initialization"""
        self.data_samples = samples
        self.targets = targets
        self.batch_size = batch_size
        self.indices = np.arange(len(samples))
        self.experiment = experiment
        self.shuffle = shuffle

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.data_samples) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of batch's sequences + targets
        samples = [self.data_samples[k] for k in indices]
        targets = [self.targets[k] for k in indices]

        # Find max limits
        if Configuration['sampling']['dynamic_batching']:
            if Configuration['model']['hierarchical']:
                max_sequences_size = max([len(sample) for sample in samples])
                max_sequence_size = max([len(sentence) for sample in samples for sentence in sample])
            else:
                max_sequences_size = 0
                max_sequence_size = max([len(sample) for sample in samples])
        else:
            max_sequences_size = Configuration['sampling']['max_sequences_size']
            max_sequence_size = Configuration['sampling']['max_sequence_size']

        # Vectorize inputs (x,y)
        x_batch, y_batch = self.experiment.encode_dataset(samples, targets,
                                                          max_sequences_size=max_sequences_size,
                                                          max_sequence_size=max_sequence_size)

        return x_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
