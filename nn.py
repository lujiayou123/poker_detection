#! /usr/bin/env python
#coding=utf-8
import numpy as np
from keras.layers import *
from keras_layer_normalization import LayerNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from nn_new import *
from util import *
from keras.initializers import Constant
from custom import *
def lstm(v_size):
    x_in = Input(shape=(None,))
       
    embedding = Embedding(v_size, emb_dim)
    e = embedding(x_in)
       
    h = LayerNormalization()(e)
    h = CuDNNLSTM(h_dim)(h)
   
    out = Dense(1, activation='sigmoid')(h)
    
    model = Model(input=x_in, output=out)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

def transformer(v_size,embedding_matrix):
    x_in = Input(shape=(None,))
    aux_in = Input(shape=(None,))

    embedding = Embedding(v_size, embed_dim,embeddings_initializer=Constant(embedding_matrix),
                                trainable=False)
    e = embedding(x_in)
    aux_e = embedding(aux_in)
    
    h = Attention(8, 16)([aux_e, e, e]) # q k v
    
    # pooling
    h = GlobalAveragePooling1D()(h)

    out = Dense(3, activation='softmax')(h)
    
    model = Model(input=[x_in,aux_in], output=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def transformer_deepwalk(g_size,embeddings_matrix):
    x_in = Input(shape=(None,))
       
    embedding = Embedding(g_size, 64, weights = [embeddings_matrix], trainable = False)
    e = embedding(x_in)
    
    h = Attention(8, 16)([e, e, e])
    
    # pooling
    h = GlobalAveragePooling1D()(h)

    out = Dense(1, activation='sigmoid')(h)
    
    model = Model(input=x_in, output=out)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

def TD_LSTM(v_size,embedding_matrix):

    input_left = Input(shape=(None,))
    input_right = Input(shape=(None,))
    embedding = Embedding(v_size, embed_dim,embeddings_initializer=Constant(embedding_matrix),trainable=False)
    text_left = SpatialDropout1D(0.2)(embedding(input_left))
    text_right = SpatialDropout1D(0.2)(embedding(input_right))
 
    # text_left = embedding(input_left)
    # text_right = embedding(input_right)

    left_x = CuDNNLSTM(h_dim)(text_left)
    right_x = CuDNNLSTM(h_dim,go_backwards=True)(text_right)
    h = concatenate([left_x,right_x],axis=-1)
    # h = GlobalAveragePooling1D()(h)
    out = Dense(3,activation='softmax')(h)

    model = Model(input=[input_left,input_right], output=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def IAN(v_size,embedding_matrix):
# epoch 8 bs=8
    input_text = Input(shape=(maxlen,))
    input_aspect = Input(shape=(asplen,))
    embedding = Embedding(v_size, embed_dim,embeddings_initializer=Constant(embedding_matrix),trainable=False)
    # text_embed = SpatialDropout1D(0.2)(embedding(input_text))
    # aspect_embed = SpatialDropout1D(0.2)(embedding(input_aspect))

    text_embed = embedding(input_text)
    aspect_embed = embedding(input_aspect)
    h_text = CuDNNLSTM(h_dim, return_sequences=True)(text_embed)
    h_aspect = CuDNNLSTM(h_dim, return_sequences=True)(aspect_embed)

    h_text_pooling = GlobalAveragePooling1D()(h_text)
    h_aspect_pooling = GlobalAveragePooling1D()(h_aspect)

    h_text_pooling = Lambda(lambda x: K.expand_dims(x,1))(h_text_pooling)
    h_aspect_pooling = Lambda(lambda x: K.expand_dims(x,1))(h_aspect_pooling)


    att_1 = Attention(8,16)([h_aspect_pooling,h_text,h_text])
    att_1 = GlobalAveragePooling1D()(att_1)

    att_2 = Attention(8,16)([h_text_pooling,h_aspect,h_aspect])
    att_2 = GlobalAveragePooling1D()(att_2)

    att = concatenate([att_1,att_2],axis=-1)
    out = Dense(3,activation='softmax')(att)

    model = Model(input=[input_text,input_aspect], output=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
  
def ATAE_LSTM(v_size,embedding_matrix):
    input_text = Input(shape=(maxlen,))
    input_aspect = Input(shape=(asplen,))
    embedding = Embedding(v_size, embed_dim,embeddings_initializer=Constant(embedding_matrix),trainable=False)
    # text_embed = SpatialDropout1D(0.2)(embedding(input_text))
    # aspect_embed = SpatialDropout1D(0.2)(embedding(input_aspect))

    text_embed = embedding(input_text)
    aspect_embed = embedding(input_aspect)
    content = concatenate([text_embed,aspect_embed],axis=1)

    h = CuDNNLSTM(h_dim, return_sequences=True)(content)
    h_a = concatenate([h,aspect_embed],axis=1)

    att = Attention(1,120)([h_a,h_a,h_a])
    att = GlobalAveragePooling1D()(att)
    att = Lambda(lambda x: K.expand_dims(x))(att)

    r = multiply([att,h_a])
    r = Lambda(lambda x: K.sum(x, axis=1))(r)
    h = Dense(3,activation='softmax')(r)
    model = Model(input=[input_text,input_aspect], output=h)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
def MemNet(v_size,embedding_matrix,use_loc_input=False):
    n_hop = 9
    input_text = Input(shape=(maxlen,))
    input_aspect = Input(shape=(1,))

    inputs = [input_text,input_aspect]

    embedding = Embedding(v_size, embed_dim,embeddings_initializer=Constant(embedding_matrix),trainable=False)
    text_embed = SpatialDropout1D(0.2)(embedding(input_text))

    #aspect_embed = embedding(input_aspect)
    aspect_embed = Flatten()(embedding(input_aspect))
    print aspect_embed

    attention_layer = Attention2(use_W=False, use_bias=True)
    linear_layer = Dense(embed_dim)
    computation_layers_out = [aspect_embed]

    for i in range(n_hop):
        # content attention layer
        repeat_out = RepeatVector(maxlen)(computation_layers_out[-1])
        concat = concatenate([text_embed, repeat_out], axis=-1)
        attend_weight = attention_layer(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        content_attend = multiply([text_embed, attend_weight_expand])
        content_attend = Lambda(lambda x: K.sum(x, axis=1))(content_attend)
        out_linear = linear_layer(computation_layers_out[-1])
        computation_layers_out.append(add([content_attend, out_linear]))

    out = Dense(3,activation='softmax')(computation_layers_out[-1])
    model = Model(input=[input_text,input_aspect], output=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model