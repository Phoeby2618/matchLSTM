
import numpy as np
import pickle
import tensorflow as tf
from batch import BatchGenerator
from dataProcess import Preprocess
from mLSTM import model

def flag():

    flags=tf.flags
    FLAGS=flags.FLAGS

    flags.DEFINE_integer('batchsize',20,'the batch_size of training procedure')
    flags.DEFINE_integer('hidden_size',150,'LSTM hidden neural size')
    flags.DEFINE_integer('embedding_size',300,'embedding dim')

    flags.DEFINE_integer('epoch',20,'num epoch')
    flags.DEFINE_integer('label_num',2,'classify labeldata num')
    flags.DEFINE_integer('vocab_size',None,'data vocabulary size')

    flags.DEFINE_integer('att_num',1,'attention num')
    flags.DEFINE_integer('att_hidden',150,'attention hidden size')

    # flags.DEFINE_string('data_dir','../dataProcess/data/SNLI/','data direcrtory')
    # flags.DEFINE_string('data_filename','train_snli.txt','data file name')
    # flags.DEFINE_string('word_dict_file','../dataProcess/word_dict/snli_3label_word_dict.pkl','word dictionary file')
    flags.DEFINE_string('data_dir','../data/prosciTail/','data direcrtory')
    flags.DEFINE_string('train_filename','scitail_train.txt','train data file name')
    flags.DEFINE_string('dev_filename','scitail_dev.txt','validata file name')
    flags.DEFINE_string('test_filename','scitail_test.txt','test file name')
    flags.DEFINE_string('word_dict_file','../data/word_dict/sciTail_word_dict.pkl','word dictionary file')
    flags.DEFINE_string('embedding_file','../data/embedding/servernew_sciTail_embedding.pkl','word embedding file')
    flags.DEFINE_string('log_dir','result/scitail/log/log1/','save_log directory')
    flags.DEFINE_string('save_models_dir','result/scitail/finaldecompmatch/save1/','finaldecompmatch checkpoint save1 directory')
    flags.DEFINE_string('train_size',None,'the number of train data ')
    flags.DEFINE_string('dev_size',None,'the number of dev data ')
    flags.DEFINE_string('test_size',None,'the number of test data ')

    return FLAGS

def train(FLAGS):

    num=int(FLAGS.train_size/FLAGS.batchsize)
    print('train data总大小：',FLAGS.train_size,'train_num:',num)
    count=0
    lr0=[0.001,0.0005,0.0003]
    ac = 0.0

    for e in range(FLAGS.epoch):
        lr=lr0[2]

        print('当前学习率：',lr,'epoch:',e)

        print('==================================================================')
        dev_gs2_loss, dev_accu, summary_test = test_epoch(FLAGS,data_valid, lr,FLAGS.dev_size)
        count += 1
        valid_writer.add_summary(summary_test, count)
        print("dev_gs2_loss:", dev_gs2_loss, "dev_accuracy", dev_accu)
        print('==================================================================')


        if  e>1 and dev_accu>ac:
            saver.save(sess, FLAGS.save_models_dir + 'trainmodel', global_step=e)
            ac=dev_accu

        for i in range(num):
            s1, s2, label, s1_len, s2_len, max_len=data_train.next_batch(FLAGS.batchsize)
            feed={models.s1:np.array(s1),models.s2:np.array(s2),models.Y:np.array(label),models.length1:np.array(s1_len),
                  models.length2:np.array(s2_len),models.lr: lr,models.input_drop:0.2,models.output_drop:1.0,
                  models.l2_regular:0.0003,models.max_len:max_len}

            gs2_loss, _ = sess.run([models.loss_gs2_label, models.gs2_opt], feed_dict=feed)
            if i%100==0:
                    summary, accu = sess.run([ merged, models.accuracy], feed_dict=feed)
                    train_writer.add_summary(summary, count)
                    print("train_epoch:%d," % e, 'total step:', count,'gs2_loss:',gs2_loss, "accuracy", accu)
            count+=1


def test_epoch(FLAGS,dataset,lr,data_size):

    num=int(data_size/FLAGS.batchsize)
    print('dev data size：', data_size,'num:', num)

    testmean_gs2_loss=0.0
    testmean_accu=0.0

    for i in range(num):
        s1, s2, label, s1_len, s2_len, max_len = dataset.next_batch(FLAGS.batchsize)
        feed = {models.s1: np.array(s1), models.s2: np.array(s2), models.Y: np.array(label),
                models.length1: np.array(s1_len), models.length2: np.array(s2_len),models.lr:lr,
                models.input_drop:1.0,models.output_drop:1.0,models.l2_regular:0.,models.max_len:max_len}
        test_gs2_loss,test_accu,summary_test= sess.run([ models.loss_gs2_label, models.accuracy,merged], feed_dict=feed)
        testmean_gs2_loss+=test_gs2_loss
        testmean_accu+=test_accu

    return testmean_gs2_loss/num,testmean_accu/num,summary_test

if __name__=='__main__':
    FLAGS=flag()
    traindata = Preprocess(FLAGS.data_dir, FLAGS.train_filename, FLAGS.word_dict_file)
    FLAGS.vocab_size = traindata.word_size
    train_sentence1 = traindata.s1
    train_sentence2 = traindata.s2
    train_label = traindata.label
    train_sen1_length = traindata.s1_length
    train_sen2_length = traindata.s2_length
    word_dict = traindata.word_dict
    FLAGS.train_size=len(train_label)

    devdata = Preprocess(FLAGS.data_dir, FLAGS.dev_filename, FLAGS.word_dict_file)
    dev_sentence1 = devdata.s1
    dev_sentence2 = devdata.s2
    dev_label = devdata.label
    dev_sen1_length = devdata.s1_length
    dev_sen2_length = devdata.s2_length
    FLAGS.dev_size = len(dev_label)

    testdata = Preprocess(FLAGS.data_dir, FLAGS.test_filename, FLAGS.word_dict_file)
    test_sentence1 = testdata.s1
    test_sentence2 = testdata.s2
    test_label = testdata.label
    test_sen1_length = testdata.s1_length
    test_sen2_length = testdata.s2_length
    FLAGS.test_size = len(test_label)


    data_train = BatchGenerator(train_sentence1, train_sentence2, train_label, train_sen1_length, train_sen2_length,
                                word_dict,True)
    data_valid = BatchGenerator(dev_sentence1, dev_sentence2, dev_label, dev_sen1_length, dev_sen2_length, word_dict,
                                False)

    data_test = BatchGenerator(test_sentence1, test_sentence2, test_label, test_sen1_length, test_sen2_length, word_dict,
                                False)

    models = model(FLAGS.hidden_size, FLAGS.batchsize, FLAGS.vocab_size, FLAGS.embedding_size, FLAGS.label_num,
                   FLAGS.att_num,FLAGS.att_hidden,FLAGS.embedding_file)

    tf.summary.scalar('gs2_loss', models.loss_gs2_label)
    tf.summary.scalar('accuracy', models.accuracy)
    merged = tf.summary.merge_all()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(FLAGS.log_dir + '/valid')
    saver = tf.train.Saver()

    train(FLAGS)



