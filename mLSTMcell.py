import tensorflow as tf

class MatchLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, input_q, max_len,state_is_tuple=True,scope=None):
        '''

        :param num_units: hidden units of match-lstm
        :param input_q:  tensor of query or premise     -->[bs,step_q,2*hiddensize]
        :param max_len:  max len of batch input_q
        :param state_is_tuple: the form of state is tuple -->(c,h)
        :param scope:
        '''
        self.H_q = input_q
        self.maxlen=max_len
        super(MatchLSTMCell, self).__init__(num_units, state_is_tuple=state_is_tuple)

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope("matchlstm"):

            W_q = tf.get_variable("W_q", shape=(2 * self._num_units, self._num_units),
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            W_p = tf.get_variable("W_p", shape=(2 * self._num_units, self._num_units),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            W_r = tf.get_variable("W_r", shape=(self._num_units, self._num_units),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b_p = tf.get_variable("b_p", shape=(self._num_units,),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            w = tf.get_variable("w", shape=(self._num_units, 1),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.get_variable("b", shape=(),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

            #compute weighted input_q

            H_q = tf.reshape(self.H_q, [-1, 2 * self._num_units])
            H_qW_q = tf.reshape(tf.matmul(H_q, W_q), [-1, self.maxlen, self._num_units])
            # shape input:[bs,2hidden]
            tempsum = tf.matmul(inputs, W_p) + tf.matmul(state[1], W_r) + b_p
            G = tf.tanh(H_qW_q + tf.tile(tf.expand_dims(tempsum,1),(1,self.maxlen,1)))
            Gw = tf.reshape(tf.matmul(tf.reshape(G, [-1, self._num_units]), w), [-1, self.maxlen])
            # weight of input_q
            alpha = tf.nn.softmax(Gw + b)
            alpha = tf.expand_dims(alpha, axis=1)
            # shape H_qalpha:[bs, 2hidden]
            H_qalpha = tf.reshape(tf.matmul(alpha, self.H_q), [-1, 2 * self._num_units])

            # gate

            W_g = tf.get_variable("W_g", shape=(2 * self._num_units, 2 * self._num_units),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            gate=tf.sigmoid(tf.matmul(H_qalpha,W_g))
            sigH_pq=tf.multiply(gate,H_qalpha)

            z = tf.concat((inputs, sigH_pq), axis=1)

        output, new_state = super(MatchLSTMCell, self).__call__(z, state, scope)
        return output, new_state
