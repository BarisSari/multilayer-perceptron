# Developed by Bayram Baris Sari
# E-mail: bayrambariss@gmail.com
# Tel No: +90 539 593 7501    

import numpy as np
import time

# parse the train data and create 2 matrices, one for inputs and the other one for outputs
train_X = []
train_Y = []
train_file = open("dataTrain.csv", 'r')
train_file.readline()  # skip the first line
train_data = [line.strip('\n').split(',') for line in train_file.readlines()]
np.random.shuffle(train_data)
for row in train_data:
        train_X.append(row[0:-1])
        train_Y.append(row[-1])
train_file.close()

train_X = np.array(train_X, dtype=int)
train_Y = np.array(train_Y, dtype=int)


def find_class(y):
    r = np.diag(np.ones(10))  # expected output
    return r[y]


def train():
    d = 64  # 64 inputs
    H = 10  # 10 hidden units
    K = 10  # 10 outputs
    w = np.zeros((H, d))  # weights in the first layer
    v = np.zeros((K, H))  # weights in the second layer
    z = np.zeros(H)       # inputs of second layer, sigmoid function of w*x
    o = np.zeros(K)       # it is necessary for softmax output,for K(=10)>2 classes
    y = np.zeros(K)       # computed output
    gradient_v = np.zeros(np.shape(v))
    gradient_w = np.zeros(np.shape(w))
    x0 = 1                # bias unit in the first layer
    z0 = 1                # bias unit in hidden unit
    cross_entropy = 0
    learning_rate = 0.015
    it = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 200]

    # set random values between -0.01 and 0.01 for v and w
    for i in range(K):
        for h in range(H):
            v[i][h] = np.random.uniform(-0.01, 0.01)
    for h in range(H):
        for j in range(d):
            w[h][j] = np.random.uniform(-0.01, 0.01)

    print("Training is starting!")
    iteration = 1
    while True:
        prev_cross_entropy = cross_entropy
        for idx, x in enumerate(train_X):
            r = find_class(train_Y[idx])
            # sigmoid function
            for h in range(H):
                p = np.dot(w[h].T, x) + x0
                # if p < -709:
                #     p = -709
                z[h] = 1/(1+np.exp(-p))
            # z = sigmoid(w*x) is the input of the second layer, o = v*z
            for i in range(K):
                o[i] = np.dot(v[i].T, z) + z0
            # we need softmax to indicate the dependence between classes
            denominator = sum(np.exp(m) for m in o)
            for i in range(K):
                y[i] = np.exp(o[i]) / denominator
            # gradient descent for weights of the second layer
            for i in range(K):
                for h in range(H):
                    gradient_v[i][h] = learning_rate*np.dot((r[i]-y[i]), z[h])
            # gradient descent for weights of the first layer
            for h in range(H):
                summ = 0
                for i in range(K):
                    summ += (r[i] - y[i]) * v[i][h]
                for j in range(d):
                    gradient_w[h][j] = learning_rate*summ*z[h]*(1-z[h])*x[j]
            # updating v, weights of the second layer
            for i in range(K):
                for h in range(H):
                    v[i][h] = v[i][h] + gradient_v[i][h]
            # updating w, weights of the first layer
            for h in range(H):
                for j in range(d):
                    w[h][j] = w[h][j] + gradient_w[h][j]
            # cross-entropy error value is updated after each x
            value = np.dot(r, np.log(y))
            cross_entropy += value
        # write the cross entropy error value for epoch=1,2,3,4,5,6,7,8,9,10,50,100,200
        if any(iteration == i for i in it):
            print("Cross-entropy error value for %d. iteration: %f" % (iteration, cross_entropy))

        # calculate the sum of the value of gradient descent matrices, after 200 epoch
        if iteration > 200:
            difference = 0
            for i in range(K):
                for h in range(H):
                    difference += np.abs(gradient_v[i][h])
            for h in range(H):
                for j in range(d):
                    difference += np.abs(gradient_w[h][j])
            # if the sum is less than the threshold or cross entropy error value is less than 200,
            # it means matrices are converging, stop training phase
            if difference < 0.00003 or prev_cross_entropy-cross_entropy < 200:
                print("Training is ended after %d iteration!" % iteration)
                break

        iteration += 1

    return w, v


def accuracy(pre, exp):
    correct = 0
    wrong = 0
    for i in range(len(exp)):
        if pre[i] == exp[i]:
            correct += 1
        else:
            wrong += 1
    acc = correct / (correct+wrong)
    print("Accuracy: (%d/%d)" % (int(correct), int(correct + wrong)),
          "{0:.3f}%".format((correct / (correct + wrong)) * 100))
    return acc


# train, and print the execution time for training
start_time = time.time()
w, v = train()
print("Training takes {0:.2f} minutes.".format((time.time() - start_time)/60))

# parse the test data and create 2 matrices, one for inputs and the other one for outputs
test_X = []
test_Y = []
test_file = open("dataTest.csv", 'r')
test_file.readline()  # skip the first line
test_data = [line.strip('\n').split(',') for line in test_file.readlines()]
for row in test_data:
        test_X.append(row[0:-1])
        test_Y.append(row[-1])
test_file.close()

test_X = np.array(test_X, dtype=int)
test_Y = np.array(test_Y, dtype=int)

print("Test is starting!")
results = []
for x in test_X:
    probability = []
    H = 10              # 10 hidden units
    K = 10              # 10 outputs
    z = np.zeros(H)     # inputs of the second layer
    o = np.zeros(K)     # outputs that necessary for softmax function
    x0 = 1              # bias unit
    z0 = 1              # bias unit
    # compute z with sigmoid function, which is the input of the second layer
    for h in range(H):
        p = np.dot(w[h].T, x) + x0
        z[h] = 1/(1+np.exp(-p))
    # o = v*z
    for i in range(K):
        o[i] = np.dot(v[i].T, z) + z0
    # compute y, i. e. output,which is equal to exp(o[i])/(sum of exp(o))
    denomm = sum(np.exp(m) for m in o)
    for i in range(K):
        probability.append(np.exp(o[i])/denomm)

    # put the index,i. e. the class label, of maximum probability
    results.append(probability.index(max(probability)))

# calculate accuracy and print it
accuracy(results, test_Y)
