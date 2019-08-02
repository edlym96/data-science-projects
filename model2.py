import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
from keras.utils import to_categorical
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

def create_one_hot_buy(row):
    if row[0] > 0:
        val = 1
    else:
        val = 0
    return val

def create_one_hot_sell(row):
    if row[0] < 0:
        val = 1
    else:
        val = 0
    return val

def create_one_hot_hold(row):
    if row[0] == 0:
        val = 1
    else:
        val = 0
    return val


def create_dataset(train, test):
    min_max_scaler =preprocessing.MinMaxScaler()
    x_train = train[['sentiment','news_volume', 'news_buzz']]
    x_train = x_train.values
    x_scaled = min_max_scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_scaled)
    x_test = test[['sentiment','news_volume', 'news_buzz']]
    x_test = x_test.values
    x_scaled = min_max_scaler.fit_transform(x_test)
    x_test = pd.DataFrame(x_scaled)
    y_train = train[['daily_return']]
    y_train['buy'] = y_train.apply(create_one_hot_buy, axis = 1)
    y_train['hold'] = y_train.apply(create_one_hot_hold, axis=1)
    y_train['sell'] = y_train.apply(create_one_hot_sell, axis=1)
    y_train.drop(y_train.columns[[0]], axis=1, inplace=True)
    y_test = test[['daily_return']]
    y_test['buy'] = y_test.apply(create_one_hot_buy, axis = 1)
    y_test['hold'] = y_test.apply(create_one_hot_hold, axis=1)
    y_test['sell'] = y_test.apply(create_one_hot_sell, axis=1)
    y_test.drop(y_test.columns[[0]], axis=1, inplace=True)
    #Convert to one_hot
    # y_train[y_train>0] = 1
    # y_train[y_train<0] = 0
    # y_test[y_test > 0] = 1
    # y_test[y_test < 0] = 0

    return x_train.values, x_test.values, y_train.values ,y_test.values

def create_model(x_train, y_train, x_test, y_test):
    train_dataset = tf.placeholder(tf.float32, shape=[None, np.shape(x_train)[1]])
    train_labels = tf.placeholder(tf.float32, shape=[None,np.shape(y_train)[1]])
    weights = tf.Variable(tf.random_uniform([np.shape(x_train)[1],np.shape(y_train)[1]], -1.0, 1.0))
    bias = tf.Variable(tf.zeros([1,np.shape(y_train)[1]]))
    logits = tf.matmul(train_dataset, weights) + bias
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=train_labels))
    train_prediction = tf.nn.softmax(logits)

    # add optimizer
    #optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    #optimizer = tf.train.AdamOptimizer().minimize(loss)
    optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)

    # Define the accuracy
    # The default threshold is 0.5, rounded off directly
    prediction = tf.round(tf.sigmoid(logits))
    # Bool into float32 type
    correct = tf.cast(tf.equal(prediction, train_labels), dtype=tf.float32)
    # Average
    accuracy = tf.reduce_mean(correct)
    # End of the definition of the model framework

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
        total_loss = 0
        iteration = 1
        for _ in range(1000):
            _, loss_value, prediction = sess.run([optimizer, loss, train_prediction], feed_dict={train_dataset: x_train, train_labels: y_train})
            total_loss += loss_value
            mean_loss = total_loss / iteration
            iteration += 1
            temp_train_acc = sess.run(accuracy, feed_dict={train_dataset: x_train, train_labels: y_train})
            temp_test_acc = sess.run(accuracy, feed_dict={train_dataset: x_test, train_labels: y_test})
            print(iteration, mean_loss, prediction)
            print(temp_train_acc, temp_test_acc)
            print(weights.eval(), bias.eval())

        return weights.eval(), bias.eval()
    # model.add(Dense(1,  # output dim is 2, one score per each class
    #                 activation='softmax',
    #                 kernel_regularizer=L1L2(l1=0.0, l2=0.1),
    #                 input_dim=np.shape(x_train)[1]))  # input dimension = number of features your data has
    # model.compile(optimizer='sgd',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))


