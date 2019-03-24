import numpy as np
import tensorflow as tf
from tqdm import trange
import logging
import matplotlib as mpl
from collections import Counter

mpl.use('Agg')  # can run on machine without display
import matplotlib.pyplot as plt
import os, re, string


class QA_base(object):
    def __init__(self, max_q_length, max_c_length, FLAGS):
        self.max_q_length = max_q_length  # all questions will be cut or padded to have max_q_length
        self.max_c_length = max_c_length
        self.FLAGS = FLAGS
        logging.getLogger().setLevel(logging.INFO)

        self.preprocessing()
        #self.unit_tests()
        self.build_model()

    def add_prediction_and_loss(self):
        raise NotImplementedError("Each Model must re-implement this method.")


    def preprocessing(self):
        """Read in the Word embedding matrix as well as the question and context paragraphs and bring them into the 
        desired numerical shape."""

        logging.info("Data preparation. This can take some seconds...")
        # load vocab
        with open(self.FLAGS.data_dir + "vocab.dat", "r") as f:
            self.vocab = f.readlines()
        self.vocab = [x[:-1] for x in self.vocab]
        # load word embedding
        if self.FLAGS.word_vec_dim == 300:
            self.WordEmbeddingMatrix = np.load(self.FLAGS.data_dir + "glove.trimmed.300.npz")['glove']
        elif self.FLAGS.word_vec_dim == 100:
            self.WordEmbeddingMatrix = np.load(self.FLAGS.data_dir + "glove.trimmed.100.npz")['glove']
        else:
            raise ValueError("word_vec_dim can be either 100 or 300")
        logging.debug("WordEmbeddingMatrix.shape={}".format(self.WordEmbeddingMatrix.shape))
        null_wordvec_index = self.WordEmbeddingMatrix.shape[0]
        # append a zero vector to WordEmbeddingMatrix, which shall be used as padding value
        self.WordEmbeddingMatrix = np.vstack((self.WordEmbeddingMatrix, np.zeros(self.FLAGS.word_vec_dim)))
        self.WordEmbeddingMatrix = self.WordEmbeddingMatrix.astype(np.float32)
        logging.debug("WordEmbeddingMatrix.shape after appending zero vector={}".format(self.WordEmbeddingMatrix.shape))

        logging.info("End data preparation.")

    def build_model(self):
        self.add_placeholders()
        print("add_placeholders")
        self.predictionS, self.predictionE, self.loss = self.add_prediction_and_loss()
        print("add_prediction_and_loss")
        self.train_op, self.global_grad_norm = self.add_training_op(self.loss)
        print("add_training_op")

    def add_placeholders(self):
        self.q_input_placeholder = tf.placeholder(tf.int32, (None, self.max_q_length), name="q_input_ph")
        self.q_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, self.max_q_length),
                                                 name="q_mask_placeholder")
        self.c_input_placeholder = tf.placeholder(tf.int32, (None, self.max_c_length), name="c_input_ph")
        self.c_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, self.max_c_length),
                                                 name="c_mask_placeholder")
        self.labels_placeholderS = tf.placeholder(tf.int32, (None, self.max_c_length), name="label_phS")
        self.labels_placeholderE = tf.placeholder(tf.int32, (None, self.max_c_length), name="label_phE")

        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout_ph")

    def add_training_op(self, loss):
        step_adam = tf.Variable(0, trainable=False)
        lr = tf.constant(self.FLAGS.learning_rate)
        if self.FLAGS.decrease_lr:
            # use adam optimizer with exponentially decaying learning rate
            rate_adam = tf.train.exponential_decay(lr, step_adam, 1, self.FLAGS.lr_d_base)
            # after one epoch: # 0.999**2500 = 0.5,  hence learning rate decays by a factor of 0.5 each epoch
            rate_adam = tf.maximum(rate_adam, tf.constant(self.FLAGS.learning_rate / self.FLAGS.lr_divider))
            # should not go down by more than a factor of 2
            optimizer = tf.train.AdamOptimizer(rate_adam)
        else:
            optimizer = tf.train.AdamOptimizer(lr)

        grads_and_vars = optimizer.compute_gradients(loss)
        variables = [output[1] for output in grads_and_vars]
        gradients = [output[0] for output in grads_and_vars]

        gradients = tf.clip_by_global_norm(gradients, clip_norm=self.FLAGS.max_gradient_norm)[0]
        global_grad_norm = tf.global_norm(gradients)
        grads_and_vars = [(gradients[i], variables[i]) for i in range(len(gradients))]

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=step_adam)
        self.optimizer = optimizer

        return train_op, global_grad_norm

    def get_feed_dict(self, batch_xc, batch_xc_mask, batch_xq, batch_xq_mask, batch_yS, batch_yE, keep_prob):
        feed_dict = {self.c_input_placeholder: batch_xc,
                     self.c_mask_placeholder: batch_xc_mask,
                     self.q_input_placeholder: batch_xq,
                     self.q_mask_placeholder: batch_xq_mask,
                     self.labels_placeholderS: batch_yS,
                     self.labels_placeholderE: batch_yE,
                     self.dropout_placeholder: keep_prob}
        return feed_dict

    
    def test(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        epochs = self.FLAGS.epochs
        for index_epoch in range(1):
#           resotring model
            saver = tf.train.Saver()
            saver.restore(sess, 'model/model')
            ############### After an epoch: evaluate on validation set ###############
            logging.info("Epoch {} finished. Doing evaluation on validation set...".format(index_epoch))
            for _ in range(1):
                with open("data/squad/vocab.dat", "r") as result:
                        vocab1 = result.readlines()
                result.close()
                st1=open(self.FLAGS.data_dir + "valC").read()
                st=st1.split(' ')
                valc=[]

                for i in range(len(st)-1):
                        s=st[i]+'\n'
                        if s in vocab1:
                            valc.append(vocab1.index(s))
                        else:
                            valc.append(2)
                valc.append(6)
                valcmask=[]
                for i in range(len(valc)):
                        valcmask.append(True)
                for i in range(400-len(valc)):
                        valc.append(115613)
                        valcmask.append(False)
                batch_xc=np.array([valc])
                batch_xc_mask=np.array([valcmask])
                st=open(self.FLAGS.data_dir + "valQ").read().split(' ')
                valc=[]

                for i in range(len(st)-1):
                        s=st[i]+'\n'
                        if s in vocab1:
                            valc.append(vocab1.index(s))
                        else:
                            valc.append(2)
                valc.append(6)
                valq=valc
                valqmask=[]
                for i in range(len(valq)):
                        valqmask.append(True)
                for i in range(30-len(valq)):
                        valq.append(115613)
                        valqmask.append(False)
                batch_xq=np.array([valq])
                batch_xq_mask=np.array([valqmask])
                valys,valye=[1],[1]
                for i in range(399):
                        valys.append(0)
                        valye.append(0)
                batch_yS=np.array([valys])
                batch_yE=np.array([valye])
                print('Before run')
                feed_dict = self.get_feed_dict(batch_xc, batch_xc_mask, batch_xq, batch_xq_mask, batch_yS,
                                               batch_yE, keep_prob=1)
                current_loss, predictionS, predictionE = sess.run([self.loss, self.predictionS, self.predictionE],
                                                                  feed_dict=feed_dict)

                print('After run')
                for k in range(list(batch_xc[0]).index(115613)):
                    print(self.vocab[batch_xc[0][k]], end=' ')
                print()
                for k in range(list(batch_xq[0]).index(115613)):
                    print(self.vocab[batch_xq[0][k]], end=' ')

                result=open("result.txt","w+")
                for k in range(predictionE[0]-predictionS[0]+1):
                    print(self.vocab[batch_xc[0][predictionS[0]+k]], end=' ')
                    result.write(st1.split(' ')[predictionS[0]+k]+' ')

    def next_batch(self, batch_size, permutation_after_epoch='None', val=False):
        if self.batch_index >= self.max_batch_index:
            # we went through one epoch. reset batch_index and initialize batch_permutation
            self.initialize_batch_processing(permutation=permutation_after_epoch)

        start = self.batch_index
        end = self.batch_index + batch_size

        Xcres = self.c_input_placeholder[self.batch_permutation[start:end]]
        Xcmaskres = self.c_mask_placeholder[self.batch_permutation[start:end]]
        Xqres = self.q_input_placeholder[self.batch_permutation[start:end]]
        Xqmaskres = self.q_mask_placeholder[self.batch_permutation[start:end]]
        yresS = self.labels_placeholderS[self.batch_permutation[start:end]]
        yresE = self.labels_placeholderE[self.batch_permutation[start:end]]
        self.batch_index += batch_size
        return Xcres, Xcmaskres, Xqres, Xqmaskres, yresS, yresE

    def initialize_batch_processing(self, permutation='None', n_samples=None):
        self.batch_index = 0
        if n_samples is not None:
            self.max_batch_index = n_samples
        if permutation == 'by_length':
            # sum over True/False gives number of words in each sample
            length_of_each_context_paragraph = np.sum(self.q_input_placeholder, axis=1)
            # permutation of data is chosen, such that the algorithm sees short context_paragraphs first
            self.batch_permutation = np.argsort(length_of_each_context_paragraph)
        elif permutation == 'random':
            self.batch_permutation = np.random.permutation(
                self.max_batch_index)  # random initial permutation
        elif (permutation == 'None' or permutation is None):  # no permutation
            self.batch_permutation = np.arange(self.max_batch_index)  # initial permutation = identity
        else:
            raise ValueError("permutation must be 'by_length', 'random' or 'None'")

    def get_f1(self, yS, yE, ypS, ypE):
        """My own, more strict f1 metric"""
        f1_tot = 0.0
        for i in range(len(yS)):
            y = np.zeros(self.max_c_length)
            s = np.argmax(yS[i])
            e = np.argmax(yE[i])
            y[s:e + 1] = 1

            yp = np.zeros_like(y)
            yp[ypS[i]:ypE[i] + 1] = 1
            yp[ypE[i]:ypS[i] + 1] = 1  # allow flipping between start and end

            n_true_pos = np.sum(y * yp)
            n_pred_pos = np.sum(yp)
            n_actual_pos = np.sum(y)
            if n_true_pos != 0:
                precision = 1.0 * n_true_pos / n_pred_pos
                recall = 1.0 * n_true_pos / n_actual_pos
                f1_tot += (2 * precision * recall) / (precision + recall)
        f1_tot /= len(yS)
        return f1_tot

    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        epochs = self.FLAGS.epochs
        batch_size = self.FLAGS.batch_size
        n_samples = len(self.labels_placeholderS)

        global_losses, global_EMs, global_f1s, global_grad_norms = [], [], [], []  # global means "over several epochs"
        SQ_global_EMs, SQ_global_f1s, SQ_EMs_val, SQ_F1s_val = [], [], [], []  # corresponding squad metrics

        for index_epoch in range(1, epochs + 1):
            progbar = trange(int(n_samples / batch_size))
            losses, ems, f1s, grad_norms = [], [], [], []
            sq_ems, sq_f1s = [], []
            self.initialize_batch_processing(permutation=self.FLAGS.batch_permutation, n_samples=n_samples)

            ############### train for one epoch ###############
            for _ in progbar:
                batch_xc, batch_xc_mask, batch_xq, batch_xq_mask, batch_yS, batch_yE = self.next_batch(
                    batch_size=batch_size, permutation_after_epoch=self.FLAGS.batch_permutation)
                feed_dict = self.get_feed_dict(batch_xc, batch_xc_mask, batch_xq, batch_xq_mask, batch_yS, batch_yE,
                                               self.FLAGS.dropout)
                _, current_loss, predictionS, predictionE, grad_norm, curr_lr = sess.run(
                    [self.train_op, self.loss, self.predictionS, self.predictionE, self.global_grad_norm,
                     self.optimizer._lr],
                    feed_dict=feed_dict)
                f1s.append(self.get_f1(batch_yS, batch_yE, predictionS, predictionE))
                losses.append(current_loss)
                grad_norms.append(grad_norm)

                if len(losses) >= 20:
                    progbar.set_postfix({'loss': np.mean(losses), 'EM': np.mean(ems), 'SQ_EM': np.mean(sq_ems), 'F1':
                        np.mean(f1s), 'SQ_F1': np.mean(sq_f1s), 'grad_norm': np.mean(grad_norms), 'lr': curr_lr})
                    global_losses.append(np.mean(losses))
                    global_EMs.append(np.mean(ems))
                    global_f1s.append(np.mean(f1s))
                    SQ_global_EMs.append(np.mean(sq_ems))
                    SQ_global_f1s.append(np.mean(sq_f1s))
                    global_grad_norms.append(np.mean(grad_norms))
                    losses, ems, f1s, grad_norms = [], [], [], []
                    sq_ems, sq_f1s = [], []