import tensorflow as tf
import pickle
import numpy as np
from encoder import Encoder


class model:
    def __init__(self,hidden_size,batchsize,vocab_size,embeding_dim,label_num,att_num,att_hidden,embedding_file):
        '''

        :param hidden_size: hidden size of match lstm
        :param batchsize:
        :param vocab_size:
        :param embeding_dim:
        :param label_num:
        :param att_num: ( match-lstm+self attention )attention num
        :param att_hidden: hidden of attention
        :param embedding_file:
        '''
        self.embedding_dim=embeding_dim
        self.vocab_size=vocab_size

        with tf.name_scope('input'):
            self.s1=tf.placeholder(dtype=tf.int32,shape=[batchsize,None],name='s1')
            self.s2=tf.placeholder(dtype=tf.int32,shape=[batchsize,None],name='s2')
            self.Y=tf.placeholder(dtype=tf.int32,shape=[batchsize],name='labeldata')
            self.length1 = tf.placeholder(dtype=tf.int32, shape=[batchsize])
            self.length2 = tf.placeholder(dtype=tf.int32, shape=[batchsize])
            self.lr=tf.placeholder(dtype=tf.float32,shape=[],name='learn_rate')
            self.input_drop=tf.placeholder(dtype=tf.float32,shape=[],name='input_dropout')
            self.output_drop = tf.placeholder(dtype=tf.float32, shape=[], name='output_dropout')
            self.l2_regular = tf.placeholder(dtype=tf.float32, shape=[], name='l2_regular')
            self.max_len=tf.placeholder(dtype=tf.int32,shape=[],name='max_len')
            tf.add_to_collection("l2_regular", self.l2_regular)

        with tf.name_scope('embedding'):
            embedding=self.embedding(embedding_file)
            sen1=tf.nn.embedding_lookup(embedding,self.s1)
            sen2=tf.nn.embedding_lookup(embedding,self.s2)

        with tf.name_scope('modeling'):

            encoder = Encoder(hidden_size,att_num,att_hidden)
            H_q = encoder.encode_preprocess(sen1, self.length1, self.input_drop,scope="premise")
            H_p = encoder.encode_preprocess(sen2, self.length2, self.input_drop,scope="hypothesis")
            H_r = encoder.encode_match(H_q, H_p, self.length2, self.input_drop,max_len=self.max_len)
            self.output=H_r

        with tf.name_scope('predict'):
            with tf.name_scope('gs2'):
                s2_w_label=self.weight_variable([hidden_size*2,label_num],name='gs2_label_weight')
                s2_b_label=tf.get_variable(name='gs2_label_baises',shape=[label_num],
                                           dtype=tf.float32,initializer=tf.constant_initializer(0.1))
                pre_gs2_label=tf.matmul(self.output,s2_w_label)+s2_b_label


        with tf.name_scope('loss' ):
            tvars = tf.trainable_variables()
            weights = [v for v in tvars if ('W_' in v.name) or('label_weight' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * self.l2_regular

            self.loss_gs2_label=tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y,logits=pre_gs2_label)
                                               + l2_loss
                                               )

        with tf.name_scope('accu'):
            correct=tf.equal(tf.cast(tf.arg_max(pre_gs2_label,1),dtype=tf.int32),tf.reshape(self.Y,[-1]))
            self.accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

        with tf.name_scope('optimizer'):
            tvars = tf.trainable_variables()

            # gs2_var=[var for var in tvars
            #             if var.name.startswith('decoder')
            #             or var.name.startswith('predict/gs2/')
            #             or var.name.startswith('encoder')]

            self.gs2_opt=self.optimizer(self.loss_gs2_label,tvars,self.lr)
            tf.add_to_collection("opt", self.gs2_opt)

    def optimizer(self,loss,train_var,lr):
        with tf.name_scope('opt'):
            grads_clips = 5.0
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_var), grads_clips)
            train_op = tf.train.AdamOptimizer(lr)
            optimizer = train_op.apply_gradients(zip(grads, train_var))

        return optimizer

    def weight_variable(self,shape,name):
        return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1),dtype=tf.float32,name=name )

    def embedding(self, embedding_file):
        with open(embedding_file,'rb') as fr:
            embed=pickle.load(fr)
        with tf.name_scope("embedding"):
            embedding=tf.convert_to_tensor(np.array(embed))
        return tf.cast(embedding,tf.float32)

if __name__=="__main__":
    m=model(50,30,37859,300,3,1,150,'../data/embedding/servernew_sciTail_embedding.pkl')