import tensorflow as tf 
from data_helper import shuffle_data, generate_batches
from model import RNN
import numpy as np 
import sys
import pickle
import pandas as pd

# Parameters
# =================================================
tf.flags.DEFINE_integer('rnn_size', 100, 'hidden units of RNN , as well as dimensionality of character embedding (default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('num_layers', 2, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 256, 'Batch Size (default : 32)')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 5, 'Number of training epochs (default: 5)')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.flags.DEFINE_string('save_dir', 'model', 'model saved directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log info directiory')
tf.flags.DEFINE_integer('save_steps', 700, 'num of train steps for saving model')
tf.flags.DEFINE_integer('num_classes', 6, 'number of classes')

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

word_id_train = np.load("word_id_train.npy")
word_id_val = np.load("word_id_val.npy")
label_train = np.load("label_train.npy")
label_val = np.load("label_val.npy")
word_embeddings = np.load("word_embeddings.npy")
word2ind = pickle.load(open("word2ind.txt", "rb"))
ind2word = pickle.load(open("ind2word.txt", "rb"))

model = RNN(rnn_size = FLAGS.rnn_size, num_layers = FLAGS.num_layers, num_classes = FLAGS.num_classes, 
            rnn_type = "gru", word_embeddings = word_embeddings)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
  sess.run(init_op)
  training_step = 0
  for epoch in range(FLAGS.num_epochs):
    training_accuracy = 0.0
    training_loss = 0.0
    word_id_train, label_train = shuffle_data(word_id_train, label_train)
  
    word_batches, label_batches, length_batches = generate_batches(word_id_train, label_train, word2ind,
                                                                  batch_size = FLAGS.batch_size)
    for batch in range(len(word_batches)):

      feed_train = {model.inputs:word_batches[batch],
                    model.targets:label_batches[batch],
                    model.learning_rate: FLAGS.learning_rate,
                    model.sequence_length:length_batches[batch],
                    model.output_keep_prob:FLAGS.dropout_keep_prob}
      _,loss, accuracy = sess.run([model.train_op, model.loss, model.accuracy], feed_dict=feed_train)
      training_accuracy = accuracy + training_accuracy
      training_loss = training_loss+ loss
      training_step += 1
      
      if (training_step%100 == 0):
        val_accuracy = 0.0
        counter = 0
        word_batch_val, label_batch_val, length_batch_val = generate_batches(word_id_val, label_val, word2ind,
                                                                  batch_size = FLAGS.batch_size)
        for i in range(len(word_batch_val)):
          feed_val = {
                      model.inputs:word_batch_val[i],
                      model.targets:label_batch_val[i],
                      model.sequence_length:length_batch_val[i],
                      model.output_keep_prob:1.0
                      }
          acc = sess.run(model.accuracy, feed_dict=feed_val)
          counter +=1
          val_accuracy = val_accuracy+acc

        training_accuracy = training_accuracy/100.0
        training_loss = training_loss/100.0
        val_accuracy = val_accuracy/float(counter)
        print('Epoch {:>3}/{} - Training Step{:>3} - Training Loss: {:>6.3f} - Training accuracy: {:>6.3f} - Validation accuracy {:>6.3f}'
               .format( epoch+1,
                        FLAGS.num_epochs,  
                        training_step,
                        training_loss,
                        training_accuracy,
                        val_accuracy
                        ))
        training_loss = 0.0
        training_accuracy = 0.0

    model_epoch = "model/model_"+str(epoch+1)+".ckpt"
    save_path = saver.save(sess, model_epoch)
    print("Model saved in path: %s" % save_path)



######generate submission #############################
# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   saver = tf.train.Saver(tf.global_variables())
#   ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
#   if ckpt and ckpt.model_checkpoint_path:
#     print ("model path : %s"%ckpt.model_checkpoint_path)
#     saver.restore(sess, ckpt.model_checkpoint_path)  
#     print ("finish training and start to make predictions on test data...")
#     sample_submission = pd.read_csv("data/sample_submission.csv")
#     list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
#     word_id_test = np.load("word_id_test.npy")
#     bsize = 128
#     word_batches_test,_, length_batches_test = generate_batches(word_id_test, [], word2ind, batch_size = bsize)
#     predictions = np.zeros((len(word_id_test), 6))
#     for i in range(len(word_batches_test)):

#       feed_test = {
#                         model.inputs:word_batches_test[i],
#                         model.sequence_length:length_batches_test[i],
#                         model.output_keep_prob:1.0
#                         }
#       prob = sess.run(model.prob, feed_dict=feed_test)
#       predictions[i*bsize:(i+1)*bsize] = prob


#     sample_submission[list_classes] = predictions

#     fn = 'submission.csv'
#     sample_submission.to_csv(fn, index=False)  
#    