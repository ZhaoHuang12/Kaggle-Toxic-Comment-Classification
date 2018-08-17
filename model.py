import tensorflow as tf 

class RNN():
  def __init__(self, rnn_size, num_layers, num_classes, rnn_type, word_embeddings):
    self.rnn_size = rnn_size
    self.num_layers = num_layers
    self.num_classes = num_classes
    self.rnn_type = rnn_type
    self.word_embeddings = word_embeddings
    self.vocab_size = word_embeddings.shape[0]
    self.embedding_dimension = word_embeddings.shape[1]

    with tf.name_scope("placeholder"):
      self.inputs = tf.placeholder(tf.int32, [None, None], name = 'inputs')
      self.targets = tf.placeholder(tf.float32, [None, self.num_classes], name = 'targets')
      self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
      self.sequence_length = tf.placeholder(tf.int32, [None,], name = 'sequence_length')
      self.output_keep_prob = tf.placeholder(tf.float32, name='out_keep_prob')

    with tf.name_scope("embedding"):
      glove_weights_initializer = tf.constant_initializer(self.word_embeddings)
      embedding_weights = tf.get_variable( name='embedding_weights', shape=(self.vocab_size, self.embedding_dimension), 
                                          initializer=glove_weights_initializer,
                                            trainable=True)
      self.embedded_inputs = tf.nn.embedding_lookup(embedding_weights, self.inputs)

    with tf.name_scope('RNN'):
      if (self.rnn_type == "lstm"): 
        cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
      else:
        cell = tf.contrib.rnn.GRUCell(self.rnn_size)        

      #add dropout
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = self.output_keep_prob)

      #stack cells
      cell = tf.nn.rnn_cell.MultiRNNCell([cell for _ in range(self.num_layers)])
      _, state = tf.nn.dynamic_rnn(cell=cell, inputs = self.embedded_inputs, sequence_length = self.sequence_length, 
                                          dtype = tf.float32)

      if (self.rnn_type=="lstm"):
        self.outputs = state[-1][1]
      else:
        self.outputs = state[-1]

    #output layer
    with tf.name_scope("score"):
      weights = tf.Variable(tf.random_normal([self.rnn_size, self.num_classes]))
      bias = tf.Variable(tf.random_normal([self.num_classes]))
      self.logits = tf.nn.bias_add(tf.matmul(self.outputs, weights), bias)
      self.prob = tf.nn.sigmoid(self.logits)


    with tf.name_scope('optimize'):
      cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=self.logits)
      self.loss = tf.reduce_sum(cross_entropy)
      trainable_vars = tf.trainable_variables()
      grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), 5.0)
      
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.train_op = optimizer.apply_gradients(zip(grads, trainable_vars))

    with tf.name_scope('accuracy'):
      correct = tf.equal(self.targets, tf.round(self.prob, name="round_probabilities"))
      casted_correct = tf.cast(correct, tf.float32, name="correct_calculation")
      self.accuracy = tf.reduce_mean(casted_correct, name="accuracy")



  def inference(self, sess, inputs, sequence_length):
    prob = sess.run(self.prob, feed_dict={self.inputs:inputs, self.sequence_length: sequence_length, self.output_keep_prob : 1.0})
    return prob

