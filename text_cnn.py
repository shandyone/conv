import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import csv_reader

#lr=0.006
input_size=30
output_size=1
batch_size=60

f=open('output3.csv')
df=pd.read_csv(f)
data=np.array(df)
train_end,test_begin=500000,500000
row=len(data[:,0])
col=len(data[0,:])
print (row,col)

def train_data():
    train_x,train_y=[],[]
    batch_index=[]
    data_train=data[:train_end,:]
    for i in range(train_end):
        if i % batch_size == 0:
            batch_index.append(i)
        x=data_train[i,:col-2]
        y=data_train[i, col-2:]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    return train_x,train_y,batch_index

def test_data():
    test_x,test_y=[],[]
    #data_test=data[test_begin:,:]
    lenth=(row-test_begin)//batch_size
    # for i in range(test_begin,col,batch_size):
    #     x=data_test[i:i+batch_size,:col-2]
    #     y=data_test[i:i+batch_size,col-2:]
    print (lenth)
    for i in range(lenth):
        x=data[test_begin+i*batch_size:test_begin+(i+1)*batch_size,:col-2]
        y=data[test_begin+i*batch_size:test_begin+(i+1)*batch_size,col-2:]
        #print (x)
        test_x.append(x.tolist())
        test_y.append(y.tolist())
    #print (test_y)
    return test_x,test_y

def weight_variable(name,shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(name,shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def conv1d(x,w):
    return tf.nn.conv1d(x,w,stride=[1,1,1],padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME')

w_conv1 = weight_variable('w_conv1',[1, 5, 1, 8])
b_conv1 = bias_variable('b_conv1',[8])
w_conv2 = weight_variable('w_conv2', [1, 5, 8, 16])
b_conv2 = bias_variable('b_conv2', [16])
w_fc1 = weight_variable('w_fc1', [19 * 1 * 16, 2])
b_fc1 = bias_variable('b_fc1', [2])#the problem I have met is the name problem, which caused by the global
def cnn(x):
    x_image = tf.reshape(x, [-1, 1, col-2, 1])
    #w_conv1 = weight_variable('w_conv1',[1, 5, 1, 8])
    #b_conv1 = bias_variable('b_conv1',[8])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)

    #w_conv2 = weight_variable('w_conv2',[1, 5, 8, 16])
    #b_conv2 = bias_variable('b_conv2',[16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)

    # keep_prob = tf.placeholder("float32")
    # h_fc1_drop = tf.nn.dropout(h_pool2, keep_prob=0.5)

    #w_fc1 = weight_variable('w_fc1',[19 * 1 * 16, 2])
    #b_fc1 = bias_variable('b_fc1',[2])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 19 * 1 * 16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    #h_fc1_dropout=tf.nn.dropout(h_fc1,keep_prob=0.5)

    return h_fc1

def train_cnn():
    X=tf.placeholder(tf.float32,shape=[None,1, col-2])
    Y=tf.placeholder(tf.float32,shape=[None,1,2])
    #global_step=tf.Variable(0,trainable=False)
    global_step=tf.Variable(tf.constant(0))
    init_global_rate = 0.006
    with tf.variable_scope("cnn"):
        y_predict=cnn(X)
    train_x,train_y,batch_index=train_data()
    print (train_y[-1])

    lr=tf.train.exponential_decay(init_global_rate,global_step=global_step,decay_steps=5,decay_rate=0.9,staircase=True)#decay every decay_steps steps with a base of decay_rate
    #if staircase is true, decay the learning rate at discrete intervals
    #cross_entropy=-tf.reduce_sum(Y*tf.log(y_predict))
    loss=tf.reduce_mean(tf.square(tf.reshape(y_predict,[-1])-tf.reshape(Y,[-1])))
    train_step=tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)
    saver=tf.train.Saver(tf.global_variables())
    #correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(Y,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    loss_=0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(200):
            for step in range(len(batch_index)-1):
                x_input=np.array(train_x[batch_index[step]:batch_index[step+1]])[:,np.newaxis,:]
                y_input=np.array(train_y[batch_index[step]:batch_index[step+1]])[:,np.newaxis,:]
                y_predict_,_,loss_=sess.run([y_predict,train_step,loss],feed_dict={X:x_input,Y:y_input})
            print(i,loss_)
            #if (i+1) % 100==0:
            #    print "save model:", saver.save(sess,'cnn.model',global_step=i)
train_cnn()

def prediction():
    X=tf.placeholder(tf.float32,shape=[None,1,col-2])
    with tf.variable_scope("cnn",reuse=True):
        test_predict=cnn(X)
    test_x,test_y=test_data()
    saver=tf.train.Saver(tf.global_variables())
    #saver = tf.train.Saver([w_conv1, b_conv1, w_conv2, b_conv2, w_fc1, b_fc1])
    predict=[]
    with tf.Session() as sess:
        module_file=tf.train.latest_checkpoint('./')
        saver.restore(sess,module_file)
        for step in range(len(test_x)):
            #test_input=np.array(test_x[step])[:,np.newaxis,:]
            test_input=np.array(test_x[step]).reshape([-1,1,col-2])
            prob=sess.run(test_predict,feed_dict={X:test_input})
            predict.append(prob.reshape([-1,2]))
        predict=np.array(predict).reshape([-1])
        test_y=np.array(test_y).reshape([-1])
        print 'predict_len:',len(predict)
        #print (test_y)
        acc=np.mean(np.abs(predict-test_y))
        #acc = int(tf.reduce_mean(tf.square(predict-test_y)))
        print "acc=:", acc
prediction()
