import os

import numpy as np
from gensim.models import KeyedVectors
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D, Conv1D, Dropout
# from tensorflow.keras.layers import CuDNNGRU, GRU
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense, Embedding, add, maximum
from tensorflow.keras.layers import Input, SpatialDropout1D, Lambda
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import multi_gpu_model
from transformers import TFAutoModel

from nomothesia_nlp.neural_networks.layers.custom_bert.load import load_google_bert
from nomothesia_nlp.data import VECTORS_DIR, MODELS_DIR
from nomothesia_nlp.experiments.configurations.configuration import Configuration
from nomothesia_nlp.neural_networks.layers import Camouflage, Attention, ContextualAttention, MultiHeadSelfAttention, BERT
from nomothesia_nlp.neural_networks.layers import TimestepDropout, SymmetricMasking, ElmoEmbeddingLayer, LayerNormalization
from nomothesia_nlp.neural_networks.neural_network import NeuralNetwork


class DocumentClassification(NeuralNetwork):
    def __init__(self, label_terms_ids):
        super().__init__()
        self._cuDNN = Configuration['task']['cuDNN']
        self._decision_type = Configuration['task']['decision_type']
        self.elmo = True if 'elmo' in Configuration['model']['token_encoding'] else False
        self.n_classes = len(label_terms_ids)
        self._attention_mechanism = Configuration['model']['attention_mechanism']
        if 'word2vec' in Configuration['model']['token_encoding']:
            self.word_embedding_path = os.path.join(VECTORS_DIR, 'word2vec', Configuration['model']['embeddings'])

    def __del__(self):
        K.clear_session()
        del self._model

    def PretrainedEmbedding(self):

        inputs = Input(shape=(None,), dtype='int32')
        embeddings = KeyedVectors.load_word2vec_format(self.word_embedding_path, binary=True)
        word_encodings_weights = np.concatenate((np.zeros((1, embeddings.syn0.shape[-1]), dtype=np.float32), embeddings.syn0), axis=0)
        embeds = Embedding(len(word_encodings_weights), word_encodings_weights.shape[-1],
                           weights=[word_encodings_weights], trainable=False)(inputs)

        return Model(inputs=inputs, outputs=embeds, name='embedding')

    def compile(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):

        shape = (Configuration['sampling']['max_sequences_size'], Configuration['sampling']['max_sequence_size'])
        if Configuration['sampling']['hierarchical']:
            if Configuration['model']['architecture'] == 'BERT':
                self._compile_hierachical_bert(shape=shape, n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size,
                                               dropout_rate=dropout_rate, word_dropout_rate=word_dropout_rate, lr=lr)
            elif isinstance(n_hidden_layers, list):
                self._compile_hans(shape=shape, n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size, dropout_rate=dropout_rate,
                                   word_dropout_rate=word_dropout_rate, lr=lr)
            else:
                self._compile_hanmax(shape=shape, n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size, dropout_rate=dropout_rate,
                                     word_dropout_rate=word_dropout_rate, lr=lr)
        elif Configuration['model']['architecture'] == 'TRANSFORMERS':
            self._compile_transformers(shape=shape, n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size, dropout_rate=dropout_rate,
                                       word_dropout_rate=word_dropout_rate, lr=lr)
        elif 'BERT' in Configuration['model']['architecture']:
            self._compile_bert(shape=shape, n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size, dropout_rate=dropout_rate,
                               word_dropout_rate=word_dropout_rate, lr=lr)
        elif Configuration['model']['attention_mechanism']:
            self._compile_bigrus_attention(shape=shape, n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size, dropout_rate=dropout_rate,
                                           word_dropout_rate=word_dropout_rate, lr=lr)
        else:
            self._compile_bigrus(n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size, dropout_rate=dropout_rate,
                                 word_dropout_rate=word_dropout_rate, lr=lr)

    def _compile_bigrus(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):
        """
        Compiles a Hierarchical RNN based on the given parameters
        :param hidden_units_size: size of hidden units, as a list
        :param dropout_rate: The percentage of inputs to dropout
        :param word_dropout_rate: The percentage of timesteps to dropout
        :param lr: learning rate
        :return: Nothing
        """

        # Document Feature Representation
        if self.elmo:
            doc_inputs = Input(shape=(1,), dtype='string', name='doc_inputs')
            doc_embs = ElmoEmbeddingLayer()(doc_inputs)
        else:
            doc_inputs = Input(shape=(None,), name='doc_inputs')
            self.pretrained_encodings = self.PretrainedEmbedding()
            doc_embs = self.pretrained_encodings(doc_inputs)

        # Apply variational dropout
        drop_doc_embs = SpatialDropout1D(dropout_rate, name='feature_dropout')(doc_embs)
        encodings = TimestepDropout(word_dropout_rate, name='word_dropout')(drop_doc_embs)

        # Bi-GRUs over token embeddings
        return_sequences = True
        for i in range(n_hidden_layers):
            if i == n_hidden_layers - 1:
                return_sequences = False
            if self._cuDNN:
                grus = Bidirectional(GRU(hidden_units_size, return_sequences=return_sequences), name='bidirectional_grus_{}'.format(i))(encodings)
            else:
                grus = Bidirectional(GRU(hidden_units_size, activation="tanh", recurrent_activation='sigmoid',
                                         return_sequences=return_sequences), name='bidirectional_grus_{}'.format(i))(encodings)
            if i != n_hidden_layers - 1:
                grus = Camouflage(mask_value=0.0)([grus, encodings])
                if i == 0:
                    encodings = SpatialDropout1D(dropout_rate)(grus)
                else:
                    encodings = add([grus, encodings])
                    encodings = SpatialDropout1D(dropout_rate)(encodings)
            else:
                encodings = grus

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(encodings)

        # Wrap up model + Compile with optimizer and loss function
        self._model = Model(inputs=doc_inputs, outputs=[outputs])
        self._model.compile(optimizer=Adam(lr=lr, clipvalue=5.0),
                            loss='binary_crossentropy' if self._decision_type == 'multi_label'
                            else 'categorical_crossentropy')

    def _compile_bigrus_attention(self, shape, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):
        """
        Compiles a Hierarchical RNN based on the given parameters
        :param hidden_units_size: size of hidden units, as a list
        :param dropout_rate: The percentage of inputs to dropout
        :param word_dropout_rate: The percentage of timesteps to dropout
        :param lr: learning rate
        :return: Nothing
        """

        # Document Feature Representation
        if self.elmo:
            doc_inputs = Input(shape=(1,), dtype='string', name='document_inputs')
            doc_embs = ElmoEmbeddingLayer()(doc_inputs)
        else:
            doc_inputs = Input(shape=(None,), name='document_inputs')
            self.pretrained_encodings = self.PretrainedEmbedding()
            doc_embs = self.pretrained_encodings(doc_inputs)

        # Apply variational dropout
        drop_doc_embs = SpatialDropout1D(dropout_rate, name='feature_dropout')(doc_embs)
        encodings = TimestepDropout(word_dropout_rate, name='word_dropout')(drop_doc_embs)

        # Bi-GRUs over token embeddings
        for i in range(n_hidden_layers):
            if self._cuDNN:
                grus = Bidirectional(GRU(hidden_units_size, return_sequences=True), name='bidirectional_grus_{}'.format(i))(encodings)
            else:
                grus = Bidirectional(GRU(hidden_units_size, activation="tanh", recurrent_activation='sigmoid',
                                         return_sequences=True), name='bidirectional_grus_{}'.format(i))(encodings)
            grus = Camouflage(mask_value=0.0)([grus, encodings])
            if i == 0:
                encodings = SpatialDropout1D(dropout_rate)(grus)
            else:
                encodings = add([grus, encodings])
                encodings = SpatialDropout1D(dropout_rate)(encodings)

        # Attention over BI-GRU (context-aware) embeddings
        if self._attention_mechanism == 'maxpooling':
            doc_encoding = GlobalMaxPooling1D(name='max_pooling')(encodings)
            losses = 'binary_crossentropy' \
                if self._decision_type == 'multi_label' else 'categorical_crossentropy'
            loss_weights = None
        elif self._attention_mechanism == 'attention':
            # Mask encodings before attention
            grus_outputs = SymmetricMasking(mask_value=0, name='masking')([encodings, encodings])
            if Configuration['model']['return_attention']:
                doc_encoding, document_attentions = Attention(kernel_regularizer=l2(), bias_regularizer=l2(),
                                                              return_attention=True, name='self_attention')(grus_outputs)
                loss = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
                losses = {'outputs': loss}
                loss_weights = {'outputs': 1.0}
            else:
                doc_encoding = Attention(kernel_regularizer=l2(), bias_regularizer=l2(), return_attention=False,
                                         name='self_attention')(grus_outputs)
                losses = 'binary_crossentropy' \
                    if self._decision_type == 'multi_label' else 'categorical_crossentropy'
                loss_weights = None

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(doc_encoding)

        # Wrap up model + Compile with optimizer and loss function
        self._model = Model(inputs=doc_inputs,
                            outputs=[outputs] if not Configuration['model']['return_attention'] else [outputs, document_attentions])
        self._model.compile(optimizer=Adam(lr=lr, clipvalue=5.0),
                            loss=losses, loss_weights=loss_weights)

    def _compile_transformers(self, shape, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):
        """
        :param hidden_units_size: size of hidden units, as a list
        :param dropout_rate: The percentage of inputs to dropout
        :param word_dropout_rate: The percentage of timesteps to dropout
        :param lr: learning rate
        :return: Nothing
        """

        # Document Feature Representation
        if self.elmo:
            doc_inputs = Input(shape=(1,), dtype='string', name='document_inputs')
            doc_embs = ElmoEmbeddingLayer()(doc_inputs)
        else:
            doc_inputs = Input(shape=(shape[1],), name='document_inputs')
            self.pretrained_encodings = self.PretrainedEmbedding()
            doc_embs = self.pretrained_encodings(doc_inputs)

        # Initial Projection layer
        doc_embs = Dense(units=hidden_units_size, activation='linear')(doc_embs)

        pos_inputs = Input(shape=(shape[1], ))

        def _get_pos_encoding_matrix(max_len: int, d_emb: int) -> np.array:
            pos_enc = np.array(
                [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
                 range(max_len)], dtype=np.float32)
            pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
            pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
            return pos_enc

        pos_embs = Embedding(shape[1]+1, doc_embs.shape[-1].value, trainable=False, name='PositionEmbedding',
                             weights=[_get_pos_encoding_matrix(shape[1]+1, doc_embs.shape[-1].value)])(pos_inputs)

        pos_embs = Camouflage(mask_value=0)([pos_embs, doc_embs])

        # Concatenate Word + Position Embeddings
        inner_inputs = add([doc_embs, pos_embs])

        # Apply variational dropout
        inner_inputs = SpatialDropout1D(dropout_rate)(inner_inputs)
        # inner_inputs = TimestepDropout(word_dropout_rate)(inner_inputs)

        # Consecutive Transformer-Encoder blocks
        for i in range(n_hidden_layers):
            document_encodings = DocumentClassification.transformer_inner_encoder(inputs=inner_inputs,
                                                                                  units=hidden_units_size,
                                                                                  dropout_rate=dropout_rate,
                                                                                  n_heads=8)
        if Configuration['model']['attention_mechanism'] != 'maxpooling':
            # Flatten Transformers outputs
            clf_h = Lambda(lambda x: K.reshape(x, (-1, K.int_shape(x)[-1])), name='flatten_timesteps')(document_encodings)
            # Define [CLS] indices (one for every BATCH_SIZE)
            cls_mask = Input(tensor=K.tf.convert_to_tensor(np.asarray([i * shape[1]
                                                                       for i in range(Configuration['model']['batch_size'])])),
                             dtype='int32',
                             name='cls_timesteps_mask')
            # Collect (Gather) [CLS] representations (outputs)
            doc_encoding = Lambda(lambda x: K.gather(x[0], K.cast(x[1], 'int32')), name='gather_cls_timesteps')([clf_h, cls_mask])
        else:
            doc_encoding = GlobalMaxPooling1D()(document_encodings)

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(doc_encoding)

        losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        # Wrap up model + Compile with optimizer and loss function
        self._model = Model(inputs=[doc_inputs, pos_inputs, cls_mask] if Configuration['model']['attention_mechanism'] != 'maxpooling'
                            else [doc_inputs, pos_inputs],
                            outputs=[outputs])
        self._model.compile(optimizer=Adam(lr=lr, clipvalue=5.0),
                            loss=losses, loss_weights=loss_weights)

    def _compile_hans(self, shape, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):
        """
        Compiles a Hierarchical Attention Network based on the given parameters
        :param shape: The shape of the sequence, i.e. (number of sections, number of tokens)
        :param hidden_units_size: size of hidden units, as a list
        :param dropout_rate: The percentage of inputs to dropout
        :param word_dropout_rate: The percentage of timesteps to dropout
        :param lr: learning rate
        :return: Nothing
        """

        # Sentence Feature Representation
        if self.elmo:
            section_inputs = Input(shape=(1,), dtype='string', name='section_inputs')
            section_embs = ElmoEmbeddingLayer()(section_inputs)
            self._features = section_embs.shape[-1].value
        else:
            section_inputs = Input(shape=(None,), name='section_inputs')
            self.pretrained_encodings = self.PretrainedEmbedding()
            section_embs = self.pretrained_encodings(section_inputs)
            self._features = section_embs.shape[-1].value

        # Apply variational dropout
        drop_section_embs = SpatialDropout1D(dropout_rate, name='feature_dropout')(section_embs)
        encodings = TimestepDropout(word_dropout_rate, name='word_dropout')(drop_section_embs)

        # Bi-GRUs over token embeddings
        for i in range(n_hidden_layers[0]):
            if self._cuDNN:
                grus = Bidirectional(GRU(hidden_units_size[0], return_sequences=True, kernel_constraint=MinMaxNorm(min_value=-2, max_value=2)), name='bidirectional_grus_{}'.format(i))(encodings)
            else:
                grus = Bidirectional(GRU(hidden_units_size[0], activation="tanh", recurrent_activation='sigmoid',
                                         return_sequences=True, kernel_constraint=MinMaxNorm(min_value=-2, max_value=2)), name='bidirectional_grus_{}'.format(i))(encodings)
            grus = Camouflage(mask_value=0.0)([grus, encodings])
            if i == 0:
                encodings = SpatialDropout1D(dropout_rate)(grus)
            else:
                encodings = add([grus, encodings])
                encodings = SpatialDropout1D(dropout_rate)(encodings)

        # Attention over BI-GRU (context-aware) embeddings
        if self._attention_mechanism == 'maxpooling':
            section_encoder = GlobalMaxPooling1D()(encodings)
        elif self._attention_mechanism == 'attention':
            if Configuration['model']['return_attention']:
                section_encoder, attention_scores = ContextualAttention(kernel_regularizer=l2(), bias_regularizer=l2(),
                                                                        return_attention=True)(encodings)
                attention_scores = Model(inputs=section_inputs, outputs=attention_scores, name='attention_scores')
            else:
                section_encoder = ContextualAttention(kernel_regularizer=l2(), bias_regularizer=l2(), return_attention=False)(encodings)

        # Wrap up section_encoder
        section_encoder = Model(inputs=section_inputs, outputs=section_encoder, name='sentence_encoder')

        # Document Input Layer
        if self.elmo:
            doc_inputs = Input(shape=(shape[0], 1,), dtype='string', name='document_inputs')
        else:
            doc_inputs = Input(shape=(shape[0], None,), name='document_inputs')

        # Distribute sentences
        if Configuration['model']['return_attention']:
            section_encodings = TimeDistributed(section_encoder, name='sentence_encodings')(doc_inputs)
            word_attentions = TimeDistributed(attention_scores, name='document_attentions')(doc_inputs)
        else:
            section_encodings = TimeDistributed(section_encoder, name='sentence_encodings')(doc_inputs)

        # Compute mask input to exclude padded sentences
        mask = Lambda(lambda x: K.sum(x, axis=2), output_shape=lambda s: (s[0], s[1]), name='find_masking')(doc_inputs)
        section_encodings = Camouflage(mask_value=0, name='camouflage')([section_encodings, mask])

        # BI-GRUs over section embeddings
        for i in range(n_hidden_layers[1]):
            if self._cuDNN:
                grus = Bidirectional(GRU(hidden_units_size[1], return_sequences=True, kernel_constraint=MinMaxNorm(min_value=-2, max_value=2)),
                                     name='bidirectional_grus_upper_{}'.format(i))(section_encodings)
            else:
                grus = Bidirectional(GRU(hidden_units_size[1], activation="tanh", recurrent_activation='sigmoid',
                                         return_sequences=True, kernel_constraint=MinMaxNorm(min_value=-2, max_value=2)),
                                     name='bidirectional_grus_upper_{}'.format(i))(section_encodings)
            grus = Camouflage(mask_value=0.0)([grus, section_encodings])
            if i == 0:
                section_encodings = SpatialDropout1D(dropout_rate)(grus)
            else:
                section_encodings = add([grus, section_encodings])
                section_encodings = SpatialDropout1D(dropout_rate)(section_encodings)

        # Attention over BI-LSTM (context-aware) sentence embeddings
        if self._attention_mechanism == 'maxpooling':
            # section_encodings = Camouflage(mask_value=0, name='masking')([section_encodings, mask])
            doc_encoding = GlobalMaxPooling1D(name='max_pooling')(section_encodings)
        elif self._attention_mechanism == 'attention':
            # section_encodings = SymmetricMasking(mask_value=0, name='masking')([section_encodings, mask])
            if Configuration['model']['return_attention']:
                doc_encoding, section_attentions = ContextualAttention(kernel_regularizer=l2(), bias_regularizer=l2(),
                                                                       return_attention=True, name='self_attention')(section_encodings)
                loss = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
                losses = {'outputs': loss}
                loss_weights = {'outputs': 1.0}
            else:
                doc_encoding = ContextualAttention(kernel_regularizer=l2(), bias_regularizer=l2(), return_attention=False, name='self_attention')(
                    section_encodings)
                losses = 'binary_crossentropy' \
                    if self._decision_type == 'multi_label' else 'categorical_crossentropy'
                loss_weights = None

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(doc_encoding)

        # Wrap up model + Compile with optimizer and loss function
        self._model = Model(inputs=doc_inputs,
                            outputs=[outputs] if not Configuration['model']['return_attention']
                            else [outputs, word_attentions, section_attentions])
        self._model.compile(optimizer=Adam(lr=lr, clipvalue=2.0),
                            loss=losses, loss_weights=loss_weights)

    def _compile_hanmax(self, shape, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):
        """
        Compiles a Hierarchical Attention Network based on the given parameters
        :param shape: The shape of the sequence, i.e. (number of sections, number of tokens)
        :param hidden_units_size: size of hidden units, as a list
        :param dropout_rate: The percentage of inputs to dropout
        :param word_dropout_rate: The percentage of timesteps to dropout
        :param lr: learning rate
        :return: Nothing
        """

        # Sentence Feature Representation
        if self.elmo:
            section_inputs = Input(shape=(1,), dtype='string', name='section_inputs')
            section_embs = ElmoEmbeddingLayer()(section_inputs)
            self._features = section_embs.shape[-1].value
        else:
            section_inputs = Input(shape=(None,), name='section_inputs')
            self.pretrained_encodings = self.PretrainedEmbedding()
            section_embs = self.pretrained_encodings(section_inputs)
            self._features = section_embs.shape[-1].value

        # Apply variational dropout
        drop_section_embs = SpatialDropout1D(dropout_rate, name='feature_dropout')(section_embs)
        encodings = TimestepDropout(word_dropout_rate, name='word_dropout')(drop_section_embs)

        # Bi-GRUs over token embeddings
        for i in range(n_hidden_layers):
            if self._cuDNN:
                grus = Bidirectional(GRU(hidden_units_size, return_sequences=True,
                                              kernel_constraint=MinMaxNorm(min_value=-2, max_value=2)), name='bidirectional_grus_{}'.format(i))(encodings)
            else:
                grus = Bidirectional(GRU(hidden_units_size, activation="tanh", recurrent_activation='sigmoid',
                                         return_sequences=True, kernel_constraint=MinMaxNorm(min_value=-2, max_value=2)
                                         ), name='bidirectional_grus_{}'.format(i))(encodings)
            grus = Camouflage(mask_value=0.0)([grus, encodings])
            if i == 0:
                encodings = SpatialDropout1D(dropout_rate)(grus)
            else:
                encodings = add([grus, encodings])
                encodings = SpatialDropout1D(dropout_rate)(encodings)

        # Attention over BI-GRU (context-aware) embeddings
        if self._attention_mechanism == 'maxpooling':
            section_encoder = GlobalMaxPooling1D()(encodings)
        elif self._attention_mechanism == 'attention':
            if Configuration['model']['return_attention']:
                encodings = SymmetricMasking()([encodings, encodings])
                section_encoder, attention_scores = ContextualAttention(kernel_regularizer=l2(), bias_regularizer=l2(),
                                                                        return_attention=True)(encodings)
                attention_scores = Model(inputs=section_inputs, outputs=attention_scores, name='attention_scores')
                loss = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
                losses = {'outputs': loss}
                loss_weights = {'outputs': 1.0}
            else:
                encodings = SymmetricMasking()([encodings, encodings])
                section_encoder = ContextualAttention(kernel_regularizer=l2(), bias_regularizer=l2(), return_attention=False)(encodings)
                losses = 'binary_crossentropy' \
                    if self._decision_type == 'multi_label' else 'categorical_crossentropy'
                loss_weights = None

        # Wrap up section_encoder
        section_encoder = Model(inputs=section_inputs, outputs=section_encoder, name='sentence_encoder')

        # Document Input Layer
        if self.elmo:
            doc_inputs = Input(shape=(shape[0], 1,), dtype='string', name='document_inputs')
        else:
            doc_inputs = Input(shape=(shape[0], None,), name='document_inputs')

        # Distribute sentences
        if Configuration['model']['return_attention']:
            section_encodings = TimeDistributed(section_encoder, name='sentence_encodings')(doc_inputs)
            word_attentions = TimeDistributed(attention_scores, name='document_attentions')(doc_inputs)
        else:
            section_encodings = TimeDistributed(section_encoder, name='sentence_encodings')(doc_inputs)

        # Compute mask input to exclude padded sentences
        mask = Lambda(lambda x: K.sum(x, axis=2), output_shape=lambda s: (s[0], s[1]), name='find_masking')(doc_inputs)
        section_encodings = SymmetricMasking(mask_value=0, name='camouflage')([section_encodings, mask])

        section_scores = TimeDistributed(Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                                               name='outputs'))(section_encodings)

        section_scores = Camouflage(mask_value=0, name='masking')([section_scores, mask])
        outputs = GlobalMaxPooling1D(name='max_pooling')(section_scores)

        # Wrap up model + Compile with optimizer and loss function
        self._model = Model(inputs=doc_inputs,
                            outputs=[outputs] if not Configuration['model']['return_attention']
                            else [outputs, word_attentions])
        self._model.compile(optimizer=Adam(lr=lr, clipvalue=2.0),
                            loss=losses, loss_weights=loss_weights)

    def _compile_hierachical_bert(self, shape, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):

        bert_encoder = DocumentClassification.bert_encoder(base_location=Configuration['model']['bert'],
                                                           max_len=None)

        # Flatten Transformers outputs
        clf_h = Lambda(lambda x: K.reshape(x, (-1, K.int_shape(x)[-1])), name='flatten_timesteps')(bert_encoder.outputs[0])

        # Define [CLS] indices (one for every BATCH_SIZE)
        cls_mask = Input(tensor=K.tf.convert_to_tensor(np.asarray([i * None
                                                                   for i in range(Configuration['model']['batch_size'])])),
                         dtype='int32',
                         name='cls_timesteps_mask')

        # Collect (Gather) [CLS] representations (outputs)
        doc_encoding = Lambda(lambda x: K.gather(x[0], K.cast(x[1], 'int32')), name='gather_cls_timesteps')([clf_h, cls_mask])

        bert_wrapper = Model(inputs=bert_encoder.inputs + [cls_mask],
                             outputs=[doc_encoding])

        # Document Input Layer
        word_inputs = Input(shape=(None, ), name='word_inputs')
        seg_inputs = Input(shape=(None,), name='seg_inputs')
        pos_inputs = Input(shape=(None,), name='pos_inputs')

        word_inputs2 = Input(shape=(None,), name='word_inputs2')
        seg_inputs2 = Input(shape=(None,), name='seg_inputs2')
        pos_inputs2 = Input(shape=(None,), name='pos_inputs2')

        word_inputs3 = Input(shape=(None,), name='word_inputs3')
        seg_inputs3 = Input(shape=(None,), name='seg_inputs3')
        pos_inputs3 = Input(shape=(None,), name='pos_inputs3')

        # Distribute sentences
        section_encodings = bert_wrapper([word_inputs, seg_inputs, pos_inputs, cls_mask])
        section_encodings2 = bert_wrapper([word_inputs2, seg_inputs2, pos_inputs2, cls_mask])
        section_encodings3 = bert_wrapper([word_inputs3, seg_inputs3, pos_inputs3, cls_mask])

        # Compute mask input to exclude padded sentences
        section_encodings = maximum([section_encodings, section_encodings2, section_encodings3])

        section_encodings = Dropout(dropout_rate)(section_encodings)

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(section_encodings)

        losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        # Wrap up model + Compile with optimizer and loss function
        model = Model(inputs=[word_inputs, word_inputs2, word_inputs3,
                              seg_inputs, seg_inputs2, seg_inputs3,
                              pos_inputs, pos_inputs2, pos_inputs3] + [cls_mask],
                      outputs=[outputs])

        self._model = multi_gpu_model(model, gpus=4)
        self._model.compile(optimizer=Adam(lr=lr),
                            loss=losses, loss_weights=loss_weights)

    def _compile_bert(self, shape, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):

        # bert_encoder = DocumentClassification.bert_encoder(base_location=Configuration['model']['bert'],
        #                                                    max_len=shape[1])
        #
        # # Flatten Transformers outputs
        # clf_h = Lambda(lambda x: K.reshape(x, (-1, K.int_shape(x)[-1])), name='flatten_timesteps')(bert_encoder.outputs[0])
        #
        # # Define [CLS] indices (one for every BATCH_SIZE)
        # cls_mask = Input(tensor=K.tf.convert_to_tensor(np.asarray([i * shape[1]
        #                                                            for i in range(Configuration['model']['batch_size'])])),
        #                  dtype='int32',
        #                  name='cls_timesteps_mask')
        #
        # # Collect (Gather) [CLS] representations (outputs)
        # doc_encoding = Lambda(lambda x: K.gather(x[0], K.cast(x[1], 'int32')), name='gather_cls_timesteps')([clf_h, cls_mask])

        word_inputs = Input(shape=(None,), name='word_inputs', dtype='int32')
        bert_encoder = TFAutoModel.from_pretrained(Configuration["model"]["bert"], from_pt=True)
        doc_encoding = K.squeeze(bert_encoder(word_inputs)[0][:, 0:1, :], axis=1)

        doc_encoding = Dropout(dropout_rate)(doc_encoding)

        # Final output (projection) layer
        losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(doc_encoding)

        # Wrap up model + Compile with optimizer and loss function
        self._model = Model(inputs=word_inputs,
                            outputs=outputs)
        self._model.compile(optimizer=Adam(lr=lr, clipvalue=5.0),
                            loss=losses, loss_weights=loss_weights)

    @staticmethod
    def transformer_inner_encoder(inputs, units=100, dropout_rate=0.2, n_heads=5):
        # ================ Transformer - Encoder ================
        # Project in 3x feature space to support Q,V,K representations
        attention = Conv1D(filters=3 * units, kernel_size=1, strides=1, padding="same")(inputs)
        attention = SymmetricMasking(mask_value=.0)([attention, inputs])
        # Multi-head Self-Attention
        attention = MultiHeadSelfAttention(n_heads=n_heads, units=units)(attention)
        attention = Camouflage(mask_value=0)([attention, inputs])
        # Convolution Projection
        attention = Conv1D(filters=units, kernel_size=1, strides=1, padding="same")(attention)
        # Apply drop-out
        attention = SpatialDropout1D(dropout_rate)(attention)
        attention = Camouflage(mask_value=0)(inputs=[attention, inputs])
        # Skip-Connection
        attention = add([inputs, attention])
        # Layer Normalization
        attention = LayerNormalization()(attention)

        # ================ PositionWiseFF ================
        # Position-wise Feed-Forward Projection
        pw_ff = TimeDistributed(Dense(units=4 * units, activation="relu"))(attention)
        pw_ff = Camouflage(mask_value=0)(inputs=[pw_ff, inputs])
        pw_ff = TimeDistributed(Dense(units=units, activation='linear'))(pw_ff)
        pw_ff = Camouflage(mask_value=0)(inputs=[pw_ff, inputs])
        # Apply drop-out
        pw_ff = SpatialDropout1D(dropout_rate)(pw_ff)
        # Skip-Connection
        pw_ff = add([attention, pw_ff])
        # Layer Normalization
        outputs = LayerNormalization()(pw_ff)

        return outputs

    @staticmethod
    def bert_encoder(base_location, max_len=512):
        return load_google_bert(base_location=os.path.join(MODELS_DIR, 'bert', base_location)+'/',
                                use_attn_mask=False,
                                max_len=max_len, verbose=False)
