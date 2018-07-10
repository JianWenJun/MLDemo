#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/6/16 上午9:34
# @Author : ComeOnJian
# @File : train_model.py

from keras.models import Sequential,Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Embedding,LSTM,Layer,initializers,regularizers,constraints,Input,\
    Dropout,concatenate,BatchNormalization,Dense,Bidirectional,\
    Concatenate,Multiply,Maximum,Subtract,Lambda,dot,Flatten,Reshape
from keras import backend as K
from sklearn.model_selection import KFold
import numpy as np


####################  模型的一些辅助类 ####################

class AttentionLayer(Layer):
    def __init__(self,step_dim,W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(AttentionLayer,self).__init__(**kwargs)

    #设置self.supports_masking = True后需要复写该方法
    def compute_mask(self, inputs, mask=None):
        return None

    #参数设置，必须实现
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    # input (None,sentence_length,embedding_size)
    def call(self, x, mask = None):
        # 计算输出
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


    def get_config(self):
        config = {'step_dim': self.step_dim}
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ManDist(Layer):
    """
    自定义定义曼哈顿距离计算层，继承Layer层，必须实现三个父类方法
    build,call,comput_output_shape
    """
    def __init__(self, **kwargs):
        self.res = None  # 表示相似度
        # self.match_vector = None
        super(ManDist, self).__init__(**kwargs)

    def build(self, input_shape):
        """Creates the layer weights.
              # Arguments
                  input_shape: Keras tensor (future input to layer)
                      or list/tuple of Keras tensors to reference
                      for weight shape computations.
              """
        super(ManDist, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.
         # Arguments
             inputs: Input tensor, or list/tuple of input tensors.
             **kwargs: Additional keyword arguments.
         # Returns
             A tensor or list/tuple of tensors.
         """
        # 计算曼哈顿距离,因为输入计算曼哈顿距离的有两个Input层分别为inputs[0]和inputs[1]
        # lstm model
        self.res  = K.exp(- K.sum(K.abs(inputs[0]-inputs[1]),axis = 1,keepdims = True))
        return self.res

        # encode_s1 = inputs[0]
        # encode_s2 = inputs[1]
        # sentence_differerce = encode_s1 - encode_s2
        # sentece_product = encode_s1 * encode_s2
        #
        # self.match_vector = K.concatenate([encode_s1,sentence_differerce,sentece_product,encode_s2],1)
        #
        # return self.match_vector

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.
               Assumes that the layer will be built
               to match that input shape provided.
               # Arguments
                   input_shape: Shape tuple (tuple of integers)
                       or list of shape tuples (one per output tensor of the layer).
                       Shape tuples can include None for free dimensions,
                       instead of an integer.

               # Returns
                   An input shape tuple.
               """
        return K.int_shape(self.res)

class ConsDist(Layer):
    """
    自定义定义曼哈顿距离计算层，继承Layer层，必须实现三个父类方法
    build,call,comput_output_shape
    """
    def __init__(self, **kwargs):
        self.res = None  # 表示相似度
        # self.match_vector = None
        super(ConsDist, self).__init__(**kwargs)

    def build(self, input_shape):
        """Creates the layer weights.
              # Arguments
                  input_shape: Keras tensor (future input to layer)
                      or list/tuple of Keras tensors to reference
                      for weight shape computations.
              """
        super(ConsDist, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.
         # Arguments
             inputs: Input tensor, or list/tuple of input tensors.
             **kwargs: Additional keyword arguments.
         # Returns
             A tensor or list/tuple of tensors.
         """
        # 计算曼哈顿距离,因为输入计算曼哈顿距离的有两个Input层分别为inputs[0]和inputs[1]
        # lstm model
        self.res = K.sum(inputs[0] * inputs[1],axis=1,keepdims=True)/(K.sum(inputs[0]**2,axis=1,keepdims=True) * K.sum(inputs[1]**2,axis=1,keepdims=True))
        return self.res
    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.
               Assumes that the layer will be built
               to match that input shape provided.
               # Arguments
                   input_shape: Shape tuple (tuple of integers)
                       or list of shape tuples (one per output tensor of the layer).
                       Shape tuples can include None for free dimensions,
                       instead of an integer.

               # Returns
                   An input shape tuple.
               """
        return K.int_shape(self.res)

class AttentionLayer1(Layer):
    def __init__(self, **kwargs):
        # self.res = None  # 表示相似度
        self.match_vector = None
        super(AttentionLayer1, self).__init__(**kwargs)

    def build(self, input_shape):
        """Creates the layer weights.
              # Arguments
                  input_shape: Keras tensor (future input to layer)
                      or list/tuple of Keras tensors to reference
                      for weight shape computations.
              """
        super(AttentionLayer1, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.
         # Arguments
             inputs: Input tensor, or list/tuple of input tensors.
             **kwargs: Additional keyword arguments.
         # Returns
             A tensor or list/tuple of tensors.
         """
        encode_s1 = inputs[0]
        encode_s2 = inputs[1]
        sentence_differerce = encode_s1 - encode_s2
        sentece_product = encode_s1 * encode_s2
        self.match_vector = K.concatenate([encode_s1,sentence_differerce,sentece_product,encode_s2],1)
        #
        return self.match_vector

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.
               Assumes that the layer will be built
               to match that input shape provided.
               # Arguments
                   input_shape: Shape tuple (tuple of integers)
                       or list of shape tuples (one per output tensor of the layer).
                       Shape tuples can include None for free dimensions,
                       instead of an integer.

               # Returns
                   An input shape tuple.
               """
        return K.int_shape(self.match_vector)

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """

    #y_t = K.cast(K.argmax(y_true,axis=1),dtype='float32')
    #y_p = K.cast(K.argmax(y_pred,axis=1),dtype='float32')
    y_t = y_true
    y_p = y_pred

    true_positives = K.sum(K.round(K.clip(y_t * y_p, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_p, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    #y_t = K.cast(K.argmax(y_true,axis=1),dtype='float32')
    #y_p = K.cast(K.argmax(y_pred,axis=1),dtype='float32')
    y_t = y_true
    y_p = y_pred

    true_positives = K.sum(K.round(K.clip(y_t * y_p, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_t, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_t, y_p, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_t, 0, 1))) == 0:
        return 0


    p = precision(y_t, y_p)
    r = recall(y_t, y_p)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def contrastive_loss(y_true,y_pred):
    """

    定义孪生网络的代价函数，对比代价函数,每个样本的误差为L=(1 - y) * d + y * max((margin - d),0) 其中margin为相似度的阈值默认为1
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    :param y_true:1表示两个样本相似，0表示不匹配,y
    :param y_pred:表示相似度d，范围是(0,1)
    :return:
    """
    margin = 0.8
    # return K.mean(y_true * K.square(y_pred) +
    #                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    return K.mean((1-y_true) * y_pred + y_true * K.maximum((margin - y_pred),0))

def create_siamese_lstm_attention_model(embedding_matrix,model_param,embedding_size = 300,max_sentence_length = 20):

    # step 1 定义孪生网络的公共层
    X = Sequential()
    embedding_layer = Embedding(
        input_dim=len(embedding_matrix,),
        output_dim=embedding_size,
        weights=[embedding_matrix],
        trainable=True,
        input_length=max_sentence_length
    )
    # 一般来说return_sequences为true时，需要使用attention
    lstm_layer = LSTM(
        units=model_param['lstm_units']
        ,return_sequences=False
    )
    # attention_layer = AttentionLayer()
    X.add(embedding_layer)
    X.add(lstm_layer)
    # X.add(attention_layer)

    #share_model为孪生网络的共同拥有的层
    share_model = X

    # step 2 模型是多输入的结构，定义两个句子的输入
    left_input = Input(shape=(max_sentence_length,), dtype='int32')
    right_input = Input(shape=(max_sentence_length,), dtype='int32')

    # Step3定义两个输入合并后的模型层
    s1_net = share_model(left_input)
    s2_net = share_model(right_input)

    # Dropout 防止过拟合连接层
    # merge_model = concatenate([s1_net,s2_net])
    # merge_model = Dropout(model_param['desen_dropout_rate'])(merge_model)
    # merge_model = BatchNormalization()(merge_model)
    #
    matching_layer = AttentionLayer1()([s1_net,s2_net])

    # merge_model = Dropout(model_param['desen_dropout_rate'])(man_layer)
    # merge_model = BatchNormalization()(merge_model)
    # # Dense层
    # activation = 'relu'
    merge_model = Dense(model_param['num_dense'])(matching_layer)
    merge_model = Dropout(model_param['desen_dropout_rate'])(merge_model)
    merge_model = BatchNormalization()(merge_model)

    # Step4 定义输出层
    output_layer = Dense(1,activation='sigmoid')(merge_model)

    model = Model(
        inputs=[left_input, right_input],
        outputs=[output_layer], name="simaese_lstm_attention"
    )
    model.compile(
        #categorical_crossentropy,contrastive_loss,binary_crossentropy
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=["accuracy",fbeta_score,precision,recall]
    )
    return model

def create_siamese_lstm_ManDistance_model(embedding_matrix,model_param,embedding_size = 300,max_sentence_length = 20):

    # step 1 定义孪生网络的公共层
    X = Sequential()
    embedding_layer = Embedding(
        input_dim=len(embedding_matrix,),
        output_dim=embedding_size,
        weights=[embedding_matrix],
        trainable=True,
        input_length=max_sentence_length
    )
    # 一般来说return_sequences为true时，需要使用attention
    lstm_layer = LSTM(
        units=model_param['lstm_units'],
        dropout=model_param['lstm_dropout_rate'],
        recurrent_dropout=model_param['lstm_re_dropout_rate']
        ,return_sequences=False
    )
    # attention_layer = AttentionLayer()

    X.add(embedding_layer)
    X.add(lstm_layer)
    # X.add(attention_layer)

    #share_model为孪生网络的共同拥有的层
    share_model = X

    # step 2 模型是多输入的结构，定义两个句子的输入
    left_input = Input(shape=(max_sentence_length,), dtype='int32')
    right_input = Input(shape=(max_sentence_length,), dtype='int32')

    # Step3定义两个输入合并后的模型层
    s1_net = share_model(left_input)
    s2_net = share_model(right_input)

    # Step4 定义输出层
    man_layer = ManDist()([s1_net,s2_net])
    out_put_layer = Dense(2, activation='softmax')(man_layer)
    # out_put_layer = Dense(1,activation='sigmoid')(man_layer)
    model = Model(
        inputs=[left_input, right_input],
        outputs=[out_put_layer], name="simaese_lstm_manDist"
    )
    model.compile(
        # contrastive_loss binary_crossentropy categorical_crossentropy
        loss= 'categorical_crossentropy',
        optimizer='adam',
        metrics=["accuracy",fbeta_score,precision,recall]
    )
    # model.predict()
    return model

def create_siamese_lstm_dssm_mdoel(embedding_matrix,embedding_word_matrix,model_param,embedding_size = 300,max_sentence_length = 20,max_word_length=25):
    # 第一部分
    # step 1 定义复杂模型的输入
    num_conv2d_layers = 1
    filters_2d = [6, 12]
    kernel_size_2d = [[3, 3], [3, 3]]
    mpool_size_2d = [[2, 2], [2, 2]]
    left_input = Input(shape=(max_sentence_length,), dtype='int32')
    right_input = Input(shape=(max_sentence_length,), dtype='int32')

    # 定义需要使用的网络层
    embedding_layer1 = Embedding(
        input_dim=len(embedding_matrix, ),
        output_dim=embedding_size,
        weights=[embedding_matrix],
        trainable=True,
        input_length=max_sentence_length
    )
    att_layer1 = AttentionLayer(20)
    bi_lstm_layer =Bidirectional(LSTM(model_param['lstm_units']))
    lstm_layer1 = LSTM(model_param['lstm_units'],
                            return_sequences=True)
    lstm_layer2 = LSTM(model_param['lstm_units'])

    # 组合模型结构,两个输入添加Embeding层
    s1 = embedding_layer1(left_input)
    s2 = embedding_layer1(right_input)


    # 在Embeding层上添加双向LSTM层
    s1_bi = bi_lstm_layer(s1)
    s2_bi = bi_lstm_layer(s2)

    # 另在Embeding层上添加双层LSTM层
    s1_lstm_lstm = lstm_layer2(lstm_layer1(s1))
    s2_lstm_lstm = lstm_layer2(lstm_layer1(s2))

    s1_lstm = lstm_layer1(s1)
    s2_lstm = lstm_layer1(s2)
    #
    cnn_input_layer = dot([s1_lstm,s2_lstm],axes=-1)
    cnn_input_layer_dot = Reshape((20,20,-1))(cnn_input_layer)
    layer_conv1 = Conv2D(filters=8,kernel_size=3,padding='same',activation='relu')(cnn_input_layer_dot)
    z = MaxPooling2D(pool_size=(2,2))(layer_conv1)

    for i in range(num_conv2d_layers):
        z = Conv2D(filters=filters_2d[i], kernel_size=kernel_size_2d[i], padding='same', activation='relu')(z)
        z = MaxPooling2D(pool_size=(mpool_size_2d[i][0], mpool_size_2d[i][1]))(z)

    pool1_flat = Flatten()(z)
    # # print pool1_flat
    pool1_flat_drop = Dropout(rate=0.1)(pool1_flat)
    ccn1 = Dense(32, activation='relu')(pool1_flat_drop)
    ccn2 = Dense(16, activation='relu')(ccn1)

    # 另在Embeding层上添加attention层
    s1_att = att_layer1(s1)
    s2_att = att_layer1(s2)

    # 组合在Embeding层上添加attention层和在Embeding层上添加双向LSTM层
    s1_last = Concatenate(axis=1)([s1_att,s1_bi])
    s2_last = Concatenate(axis=1)([s2_att,s2_bi])

    cos_layer = ConsDist()([s1_last,s2_last])
    man_layer = ManDist()([s1_last,s2_last])
    # 第二部分
    left_w_input = Input(shape=(max_word_length,), dtype='int32')
    right_w_input = Input(shape=(max_word_length,), dtype='int32')

    # 定义需要使用的网络层
    embedding_layer2 = Embedding(
        input_dim=len(embedding_word_matrix, ),
        output_dim=embedding_size,
        weights=[embedding_word_matrix],
        trainable=True,
        input_length=max_word_length
    )
    lstm_word_bi_layer = Bidirectional(LSTM(6))
    att_layer2 = AttentionLayer(25)

    s1_words = embedding_layer2(left_w_input)
    s2_words = embedding_layer2(right_w_input)

    # s1_word_lstm = lstm_layer1(s1_words)
    # s2_word_lstm = lstm_layer1(s2_words)
    #
    # cnn_input_layer1 = dot([s1_word_lstm, s2_word_lstm], axes=-1)
    # cnn_input_layer_dot1 = Reshape((25, 25, -1))(cnn_input_layer1)
    # layer_conv11 = Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')(cnn_input_layer_dot1)
    # z1 = MaxPooling2D(pool_size=(2, 2))(layer_conv11)
    #
    # for i in range(num_conv2d_layers):
    #     z1 = Conv2D(filters=filters_2d[i], kernel_size=kernel_size_2d[i], padding='same', activation='relu')(z1)
    #     z1 = MaxPooling2D(pool_size=(mpool_size_2d[i][0], mpool_size_2d[i][1]))(z1)
    #
    # pool1_flat1 = Flatten()(z1)
    # # print pool1_flat
    # pool1_flat_drop1 = Dropout(rate=0.1)(pool1_flat1)
    # mlp11 = Dense(32, activation='relu')(pool1_flat_drop1)
    # mlp21 = Dense(16, activation='relu')(mlp11)

    s1_words_bi = lstm_word_bi_layer(s1_words)
    s2_words_bi = lstm_word_bi_layer(s2_words)

    s1_words_att = att_layer2(s1_words)
    s2_words_att = att_layer2(s2_words)

    s1_words_last = Concatenate(axis=1)([s1_words_att,s1_words_bi])
    s2_words_last = Concatenate(axis=1)([s2_words_att,s2_words_bi])
    cos_layer1 = ConsDist()([s1_words_last,s2_words_last])
    man_layer1 = ManDist()([s1_words_last,s2_words_last])


    # 第三部分，前两部分模型组合
    s1_s2_mul = Multiply()([s1_last,s2_last])
    s1_s2_sub = Lambda(lambda x: K.abs(x))(Subtract()([s1_last,s2_last]))
    s1_s2_maxium = Maximum()([Multiply()([s1_last,s1_last]),Multiply()([s2_last,s2_last])])
    s1_s2_sub1 = Lambda(lambda x: K.abs(x))(Subtract()([s1_lstm_lstm,s2_lstm_lstm]))


    s1_words_s2_words_mul = Multiply()([s1_words_last,s2_words_last])
    s1_words_s2_words_sub = Lambda(lambda x: K.abs(x))(Subtract()([s1_words_last,s2_words_last]))
    s1_words_s2_words_maxium = Maximum()([Multiply()([s1_words_last,s1_words_last]),Multiply()([s2_words_last,s2_words_last])])

    last_list_layer = Concatenate(axis=1)([s1_s2_mul,s1_s2_sub,s1_s2_sub1,s1_s2_maxium,s1_words_s2_words_mul,s1_words_s2_words_sub,s1_words_s2_words_maxium])
    last_list_layer = Dropout(0.05)(last_list_layer)
    # Dense 层
    dense_layer1 = Dense(32,activation='relu')(last_list_layer)
    dense_layer2 = Dense(48,activation='sigmoid')(last_list_layer)

    output_layer = Concatenate(axis=1)([dense_layer1,dense_layer2,cos_layer,man_layer,cos_layer1,man_layer1,ccn2])
    # Step4 定义输出层
    output_layer = Dense(1, activation='sigmoid')(output_layer)

    model = Model(
        inputs=[left_input,right_input,left_w_input,right_w_input],
        outputs=[output_layer], name="simaese_lstm_attention"
    )
    model.compile(
        # categorical_crossentropy,contrastive_loss,binary_crossentropy
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=["accuracy", fbeta_score, precision, recall]
    )
    return model

def predict(model,X_s1,X_s2):

    # y1 = model.predict([X_s1,X_s2]).reshape(-1)
    # y2 = model.predict([X_s1,X_s2]).reshape(-1)

    y1 = model.predict([X_s1,X_s2])
    y2 = model.predict([X_s1,X_s2])
    print y1.shape
    res = (y1 + y2)/2
    # print res[0:15]
    return res

def predict1(model,X_s1,X_s2,X_s1_char,X_s2_char):

    # y1 = model.predict([X_s1,X_s2]).reshape(-1)
    # y2 = model.predict([X_s1,X_s2]).reshape(-1)
    y1 = model.predict([X_s1,X_s2,X_s1_char,X_s2_char])
    y2 = model.predict([X_s1,X_s2,X_s1_char,X_s2_char])
    res = (y1 + y2)/2
    # print res[0:15]
    return res

#################### Stacking 模型的融合 ####################
from sklearn.naive_bayes import GaussianNB
# import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class StackingBaseClassifier(object):

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """
        主要用于Stacking的基础模型的训练
        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :return:model和auc
        """
        pass

    def predict(self, model, x_test):
        pass

    def get_model_out(self, x_train, y_train, x_test, n_fold=5):
        """
        交叉验证预测出基础模型的输出
        :param x_train:
        :param y_train:
        :param x_test:
        :param n_fold:
        :return:
        """
        n_train = x_train.shape[0]
        n_test = x_test.shape[0]

        train_oofp = np.zeros((n_train,))  # 存储每个fold的预测结果
        test_oofp = np.zeros((n_test, n_fold))  # 存储对测试集预测结果

        kfold = KFold(n_splits=n_fold, random_state=44, shuffle=True)


        for index, (ix_train, ix_val) in enumerate(kfold.split(x_train)):
            print '{} fold of {} start train and predict...'.format(index, n_fold)
            X_fold_train = x_train[ix_train]
            y_fold_train = y_train[ix_train]

            X_fold_val = x_train[ix_val]
            y_fold_val = y_train[ix_val]

            model = self.train(X_fold_train, y_fold_train, X_fold_val, y_fold_val)

            train_oofp[ix_val] = self.predict(model, X_fold_val)
            test_oofp[:, index] = self.predict(model, x_test)

        test_oofp_mean = np.mean(test_oofp, axis=1)
        return train_oofp, test_oofp_mean

class GussianNBClassifier(StackingBaseClassifier):
    def __init__(self):
        # 参数设置
        pass

    def train(self, x_train, y_train, x_val, y_val):
        print 'use GaussianNB train model...'
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        return gnb #, gnb.score(x_val, y_val)

    def predict(self, model, x_test):
        print 'use GaussianNB model test... '
        return model.predict(x_test)

# class LGBClassifier(StackingBaseClassifier):
#     def __init__(self):
#         self.lgb_param = {
#             'objective': 'binary',
#             'metric': {'auc', 'binary_logloss'},
#             'boosting': 'gbdt',
#             'device': 'cpu',
#             'feature_fraction': 0.8,  # 抽取所有特征的0.75个进行训练
#             'num_leaves': 16,
#             'learning_rate': 0.01,
#             'verbose': 1,
#             'bagging_seed': 456,
#             'feature_fraction_seed': 456
#         }
#
#     def train(self, x_train, y_train, x_val, y_val):
#         print 'use LGB train model...'
#         lgb_data_train = lgb.Dataset(x_train, y_train)
#         lgb_data_val = lgb.Dataset(x_val, y_val)
#         evals_res = {}
#
#         model = lgb.train(
#             params=self.lgb_param,
#             train_set=lgb_data_train,
#             valid_sets=[lgb_data_train, lgb_data_val],  # 训练集和测试集都需要验证
#             valid_names=['train', 'val'],
#             evals_result=evals_res,
#             num_boost_round=2500,
#             early_stopping_rounds=10,
#             verbose_eval=False
#         )
#         return model
#
#     def predict(self, model, x_test):
#         print 'use LGB model test... '
#         return model.predict(x_test)

class RFClassifer(StackingBaseClassifier):
    def train(self, x_train, y_train, x_val, y_val):
        print 'use RandomForest train model...'
        clf = RandomForestClassifier(n_estimators=25,
                                     max_depth=4,
                                     class_weight={
                                         0: 1,
                                         1: 4
                                     }
                                     )
        clf.fit(x_train, y_train)
        return clf #, 0.

    def predict(self, model, x_test):
        print 'use RandomForest test...'
        return model.predict(x_test)

class LogisicClassifier(StackingBaseClassifier):
    def train(self, x_train, y_train, x_val=None, y_val=None):
        print 'use LogisticRegression train model...'
        lr = LogisticRegression(class_weight={0: 1, 1: 4})
        lr.fit(x_train, y_train)
        return lr
    def predict(self, model, x_test):
        print 'use LogisticRegression test...'
        return model.predict(x_test)

class DecisionClassifier(StackingBaseClassifier):
    def train(self, x_train, y_train, x_val=None, y_val=None):
        print 'use DecisionClassifier train model...'
        dt = DecisionTreeClassifier(class_weight={0: 1, 1: 4},max_depth=5)
        dt.fit(x_train, y_train)
        return dt
    def predict(self, model, x_test):
        print 'use DecisionClassifier test...'
        return model.predict(x_test)







