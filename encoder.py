import tensorflow as tf
from mLSTMcell import MatchLSTMCell


class Encoder(object):
    def __init__(self, hidden_size,att_num,att_hidden):
        self.hidden_size = hidden_size
        self.att_num=att_num
        self.att_hidden=att_hidden

    def encode_preprocess(self, inputs, masks, keep_prob, scope="", reuse=False): #wat is masks

        self.cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True),
                                                  input_keep_prob=keep_prob)
        with tf.variable_scope(scope):
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cell,
                                                                  self.cell,
                                                                  inputs,
                                                                  sequence_length=masks,
                                                                  dtype=tf.float32)

        return tf.concat((out_fw, out_bw), 2)

    def encode_match(self, input_q, input_p, masks_p, keep_prob, max_len, scope="",reuse=False):
        self.match_cell = tf.contrib.rnn.DropoutWrapper(MatchLSTMCell(self.hidden_size,input_q,max_len,state_is_tuple=True),
                                                        input_keep_prob=keep_prob)
        with tf.variable_scope(scope):
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.match_cell,
                                                                  self.match_cell,
                                                                  input_p,
                                                                  sequence_length=masks_p,
                                                                  dtype=tf.float32)

        # get attentive sentence vector
        output=tf.concat([out_fw,out_fw],axis=2)
        attention1 = self.attention_new(self.att_num, self.att_hidden, output, self.hidden_size * 2,'att1',reuse=False)
        att_out=tf.reshape(tf.matmul(attention1, output),[-1,self.hidden_size * 2 ])

        return att_out


    # self attenion function
    def attention_new(self,att_num, att_hidden, out ,size,scope,reuse):
        with tf.variable_scope(scope,reuse=reuse):
            ws1 = tf.Variable(tf.truncated_normal(shape=[att_hidden,size],stddev=0.1),dtype=tf.float32,
                              name="att_weight_s1" )
            ws2 = tf.Variable(tf.truncated_normal(shape=[att_num, att_hidden], stddev=0.1), dtype=tf.float32,
                              name="atten_weight_s2")
        att = tf.map_fn(
            lambda sen: tf.matmul(ws2, sen),
            tf.tanh(
                tf.map_fn(
                    lambda sentence: tf.matmul(ws1, tf.transpose(sentence)), out, dtype=tf.float32)
            ),
            dtype=tf.float32)

        #imple mask 0
        my_softmax = tf.map_fn(lambda t: self.get_softmax(t), att, dtype=tf.float32)

        return my_softmax

    def get_softmax(self,T):

        condition_mask = tf.cast(~tf.equal(T, 0.), tf.int32)
        partitioned_T = tf.dynamic_partition(T, condition_mask, 2)
        # Applying the operation to the target partition:
        partitioned_T[1] = tf.nn.softmax(partitioned_T[1])

        # Stitching back together, flattening T and its indices to make things easier::
        condition_indices = tf.dynamic_partition(tf.range(tf.size(T)), tf.reshape(condition_mask, [-1]), 2)
        res_T = tf.dynamic_stitch(condition_indices, partitioned_T)
        res_T = tf.reshape(res_T, tf.shape(T))
        return res_T
