#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/7/24 上午10:25 
# @Author : ComeOnJian 
# @File : main.py 
# 参考: https://github.com/rockingdingo/deepnlp/blob/r0.1.6/deepnlp/textsum/README.md

import numpy as np
from NLP.Seq2Seq.text_summarizer import *

# seq2seq相关的配置
class ModelLoader(object):
    def __init__(self):
        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Initializing text summarization class...")
        self.model = self._init_model(self.session)

    def _init_model(self,session):
        model = create_model(session, True)
        return model

    def func_predict(self,sentence):
        """
        输入语句进行预测
        :param sentence:要取摘要的语句
        :return:
        """
        vocab_path = os.path.join(FLAGS.data_dir, "vocab")
        vocab,rev_vocab = data_util.initialize_vocabulary(vocab_path)

        token_ids = data_util.sentence_to_token_ids(sentence,vocab,tokenizer=data_util.jieba_tokenizer)
        print (token_ids)
        # 该句子对应的bucket
        bucket_id = min([b for b in xrange(len(buckets))
                         if buckets[b][0] > len(token_ids)])

        print ("current bucket id" + str(bucket_id))
        # feed data
        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)

        _, _, output_logits_batch = self.model.step(self.session,encoder_inputs,decoder_inputs,target_weights,bucket_id,True)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        output_logits = []
        for item in output_logits_batch:
            output_logits.append(item[0])
        outputs = [int(np.argmax(logit)) for logit in output_logits]
        if data_util.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_util.EOS_ID)]

        result_str = " ".join([rev_vocab[output] for output in outputs])
        return result_str

if __name__ == '__main__':
    model = ModelLoader()
    print model.func_predict('中国矿业大学是一个很值得去的大学。。。。啊哈哈哈')

