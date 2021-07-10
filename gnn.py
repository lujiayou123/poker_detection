#! /usr/bin/env python
#coding=utf-8
from keras.layers import *
from keras_layer_normalization import LayerNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from nn_new import *
from util import *
from keras.initializers import Constant
from kegra.layers.graph import GraphConvolution
from kegra.utils import *

def gcn(N,):
    x_in = Input(shape=(maxlen,embed_dim ))
    a_in = Input(shape=(maxlen,embed_dim ))
    g = Input((batch_size,), sparse=True)

    x = x_in
    a = a_in

    h = Attention(8, 16)([a, x, x])
    h = GlobalAveragePooling1D()(h)

    # gcn, support=1
    h = GraphConvolution(h_dim, 1, activation='relu')([h,g])
    #h = Dropout(0.2)(h)
    #out = GraphConvolution(3, 1, activation='softmax')([h,g])
    out = Dense(3, activation='softmax')(h)

    model = Model(inputs=[x_in,a_in,g], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def gcn_bp(F,N,n_classes):
    x_in = Input(shape=(1,embed_dim ))
    g = Input((batch_size,), sparse=True)
          
    # transformer
    # embedding = Embedding(v_size, embed_dim,embeddings_initializer=Constant(embed_matrix),trainable=False)
    # embedding = Embedding(v_size, embed_dim)
    # e = embedding(x_in)
    # e = SpatialDropout1D(0.2)(e)

    e = x_in
    h = Attention(8, 16)([e, e, e])
    h = GlobalAveragePooling1D()(h)
    #
    # # gcn, support=1
    # h = Dropout(0.5)(h)
    # h = GraphConvolution(h_dim, 1, activation='relu')([h,g])
    # h = Dropout(0.2)(h)

    out = Dense(3, activation='softmax')(h)
    #
    #

    model = Model(inputs=[x_in,g], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def gcn_chebyshev(v_size,F,N,n_classes,support):
    x_in = Input(shape=(F, ))
    g = [Input((N, ), sparse=True) for i in range(support)]
          
    # transformer
    embedding = Embedding(v_size, emb_dim)
    e = embedding(x_in)
    h = Attention(8, 16)([e, e, e])
    h = GlobalAveragePooling1D()(h)

    # gcn, support=1
    out = GraphConvolution(n_classes, support, activation='sigmoid')([h]+g)

    model = Model(inputs=[x_in]+g, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model
