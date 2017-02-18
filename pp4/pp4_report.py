###############################################################################
#   Programming Project 4 for Tufts Comp 136
#   Date: Dec 4, 2016
#   Author: Hongyan Wang
###############################################################################

###############################################################################
#   Import necessary libraries
###############################################################################
import numpy as np
import random
import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import warnings
#   Turn off warnings
warnings.filterwarnings("ignore")

###############################################################################
#   Read in data
###############################################################################

def readData():
    """
    There are 200 documents in folder 20newsgroups and one index.csv file 
    contains their corresponding labels 
    """
    data = []
    for i in range(200):
        f = open("./20newsgroups/" + str(i + 1))
        for row in f:
            data.append(row.strip().split(' '))
        f.close()

    labels = []
    f = open("./20newsgroups/index.csv")
    for row in f:
        labels.append(float(row.strip().split(',')[1]))
    f.close()

    artificial = []
    for i in range(10):
        f = open("./artificial/" + str(i + 1))
        for row in f:
            artificial.append(row.strip().split(' '))
        f.close()

    return data, labels, artificial



###############################################################################
#   Task 1: Gibbs Sampling
###############################################################################

def GibbsSampler(K, alpha, beta, corpus, num_iters):
    """
    Input: K is the number of topics
           alpha is dirichlet parameter for the topic distribution
           beta is the dirichlet parameter for the word distribution
           corpus is a list of document, each document contains some words
    return: topic indices, topiccountsperdocument, word counts per topic
    """

    # set random seed 
    random.seed(1)

    # D is the number of documents
    D = len(corpus)

    # N_words is the total number of words in corpus
    N_words = 0

    # w_indices[i] means the word index in the vocabulary of ith word in corpus
    # d_indices[i] means the document index of ith word in corpus
    # z_indices[i] means the topic index of ith word in corpus 
    w_indices     = []
    d_indices = []
    z_indices    = []

    # vocabulary contains the count of each word
    # vocabulary_indices contain the word index of each word
    # indices_to_words contains a list of words, it's the true vocabulary
    vocabulary = {}
    vocabulary_indices = {}
    indices_to_words = []

    # build w_indices, d_indices, z_indices(randomly)
    # build vocabulary, vocabulary_indices, indices_to_words
    word_index = 0
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            
            N_words += 1
            if(vocabulary.get(corpus[i][j], 0.0) > 0.0):
                w_indices.append(vocabulary_indices[corpus[i][j]])
            else: 
                vocabulary_indices[corpus[i][j]] = word_index
                indices_to_words.append(corpus[i][j])
                w_indices.append(word_index)
                word_index += 1
            
            vocabulary[corpus[i][j]] = vocabulary.get(corpus[i][j], 0.0) + 1.0

            d_indices.append(i)
            z_indices.append(random.randint(0, K - 1))
    
    w_indices = np.array(w_indices)
    d_indices = np.array(d_indices)
    z_indices = np.array(z_indices)

    # V is the lengh of the true vocabulary
    V = len(vocabulary.keys())

    # Step 1: generate a random permutation of {0,1,2,..., num_words-1}
    pi_permu = np.random.permutation(N_words)

    # Step 2: Initialize a D by K matrix of topic counts per document C_d
    C_d = np.zeros((D,K))
    # Step 3: Initialize a K by V matrix of word counts per topic C_t
    C_t = np.zeros((K,V))

    # build C_d and C_t
    for i in range(N_words):
        C_d[d_indices[i]][z_indices[i]] += 1
        C_t[z_indices[i]][w_indices[i]] += 1

    #Step 4: Initialize a 1 by K array of probabilities P (to zero)
    P = np.ones(K)
    for k in range(K):
        P[k] = 0

    # Start iterations
    for i in range(num_iters):
        for n in range(N_words):

            word = w_indices[pi_permu[n]]
            doc = d_indices[pi_permu[n]]
            topic = z_indices[pi_permu[n]]
            C_d[doc][topic] = C_d[doc][topic] - 1
            C_t[topic][word] = C_t[topic][word] - 1    

            for k in range(K):
                
                tmp_sum1 = np.sum(C_t[k, :])
                tmp_sum2 = np.sum(C_d[doc, :])
                
                P[k] = (C_t[k][word] + beta) * (C_d[doc][k] + alpha) / ((V * beta + tmp_sum1) * (K * alpha + tmp_sum2))
            
            # Normalize P
            P_sum = np.sum(P)
            for k in range(K):
                P[k] = P[k] * 1.0 /P_sum
            

            # Sample from P
            random_num = np.random.random_sample()
            topic = 0
            tmp_sum = 0.0
            for k in range(K):
                tmp_sum += P[k]
                if(random_num < tmp_sum):
                    break
                else:
                    topic += 1

            # update z_indices, C_d, C_t
            z_indices[pi_permu[n]] = topic
            C_d[doc][topic] = C_d[doc][topic] + 1
            C_t[topic][word] = C_t[topic][word] + 1

    return z_indices, C_d, C_t, indices_to_words

def WriteFrequentWords(K, C_t, vocabulary):
    V = len(vocabulary)

    with open('topicwords.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for k in range(K):
            words_frequencies = []
            for v in range(V):
                words_frequencies.append((C_t[k][v], vocabulary[v]))
            words_frequencies = sorted(words_frequencies, reverse= True)[0:5]
            spamwriter.writerow([item[1] for item in words_frequencies])

##############################################################################
#   Task 2: Classification
##############################################################################


def getTopicRepresentation(K, alpha, C_d, corpus):
    print "get topic representation..."
    rep = []
    for d in range(len(corpus)):
        
        tmp = []
        tmp_sum = np.sum(C_d[d, :])
        for k in range(K):
            tmp.append((C_d[d][k] + alpha)/(K * alpha + tmp_sum))
        rep.append(tmp)
    return rep

def getBagOfWords(corpus, vocabulary):
    print "get bag of words representation..."
    rep = []
    V = len(vocabulary)
    for d in range(len(corpus)):
        tmp_dic = {}
        for i in range(len(corpus[d])):
            tmp_dic[corpus[d][i]] = tmp_dic.get(corpus[d][i],0.0) + 1.0

        tmp = []
        for i in range(V):
            tmp.append(tmp_dic.get(vocabulary[i],0.0) * 1.0 / len(corpus[d]))
        rep.append(tmp)
    return rep

###############################################################################
# functions split_train_test(), sigmoid(), NewtonMethod(), getAccuracy(), 
# getLearningCurve() are all copied from Programming Project 3
###############################################################################
def split_train_test(dataset, label, ratio):
    """
    Input: dataset, size m by n, label, size m by 1, ratio is a float number
    Output: randomly split dataset and label into two parts, fraction ration is 
            trainning, the remaining is for test
    """
    combined_data = zip(dataset,label)
    random.shuffle(combined_data)
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    boundary = int(len(dataset) * ratio)
    for i in range(boundary):
        train_data.append(combined_data[i][0])
        train_label.append(combined_data[i][1])
    for j in range(boundary, len(dataset)):
        test_data.append(combined_data[j][0])
        test_label.append(combined_data[j][1])
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    return train_data, train_label, test_data, test_label

def sigmoid(a):
    if(a > 15):
        return 0.99999999
    elif(a < -15):
        return 0.00000001
    else:
        return 1.0 / (1 + np.exp(-a))

def NewtonMethod(train_data, train_label):
    
    alpha = 0.01
    
    new_train_data = np.ones((len(train_data), 1), dtype = float)
    new_train_data = np.append(new_train_data, train_data, axis = 1)

    num_features = len(new_train_data[0])

    w_prev = np.zeros(num_features)

    
    y = np.array([sigmoid(np.dot(w_prev, new_train_data[i])) for i in range(len(new_train_data))])

    R = np.diag(np.multiply(y, 1- y))
    

    part1 = np.linalg.inv(alpha * np.identity(num_features) + np.dot(np.dot(new_train_data.transpose(), R), new_train_data))
    part2 = np.dot(new_train_data.transpose(), (y - train_label)) + alpha * w_prev

    w_next = w_prev - np.dot(part1, part2)


    count = 0

    while((count == 0) or ((count <= 100) and (np.linalg.norm(w_next - w_prev)/np.linalg.norm(w_prev) >= 0.001))):
        
        w_prev = w_next

        y = np.array([sigmoid(np.dot(w_prev, new_train_data[i])) for i in range(len(new_train_data))])
        R = np.diag(np.multiply(y, 1- y))
        
        part1 = np.linalg.inv(alpha * np.identity(num_features) + np.dot(np.dot(new_train_data.transpose(), R), new_train_data))
        part2 = np.dot(new_train_data.transpose(), (y - train_label)) + alpha * w_prev
        
        w_next = w_prev - np.dot(part1, part2)
        count += 1
    
    w0 = w_next[0]
    w = np.array(w_next[1:])
    return w0, w

def getAccuracy(w0, w, test_data, test_label):
    """
    Input: w0 and w are parameters from training data
    Return: predict accuracy on test dataset. 
    """
    accu = 0.0
    for i in range(len(test_label)):
        predit = sigmoid(np.dot(w, test_data[i]) + w0) > 0.5
        if((predit == True and test_label[i] == 1.0) or (predit == False and test_label[i] == 0.0)):
            accu += 1.0
    return accu / len(test_label)

def getLearningCurve(data, label, flag, ratio = 2.0/3.0, random_times = 30):

    if(flag == 1):
        print "Generate learning curves for topic representation..."
    else:
        print "Generate learning curves for bags of words representation..."

    learnCurve = []
    train_sizes = np.arange(0.05,1.05,0.05)

    for dummy in range(random_times):
        print "Iteration " + str(dummy + 1) + " ... "

        train_data, train_label, test_data, test_label = split_train_test(data, label, ratio)
        
        tmp_perf = []
        for train_size in train_sizes:
            real_train_data, real_train_label, dummy_test_data, dummy_test_label = split_train_test(train_data, train_label, train_size)
                      
            w0, w = NewtonMethod(real_train_data, real_train_label)
            
            tmp_perf.append(getAccuracy(w0, w, test_data, test_label))

        learnCurve.append(np.array(tmp_perf))
    
    learnCurve = np.array(learnCurve)
    learnCurve_mean = np.mean(learnCurve, axis = 0)
    learnCurve_std = np.std(learnCurve, axis = 0)
    # for i in range(len(learnCurve_std)):
    #     learnCurve_std[i] = learnCurve_std[i] * 1.0 / len(learnCurve)

    return learnCurve_mean, learnCurve_std, len(label) * ratio * train_sizes



if __name__ == "__main__":
    print "**********************************************************************"
    print "read the data..."
    # newsgroups_data contains 200 documents, and each document is a list of words
    # newsgroups_labels contains 200 float numbers 0 or 1, correspoding 200 documents
    # artificial_data contains 10 documents, each document is a list of words
    newsgroups_data, newsgroups_labels, artificial_data = readData()

    print "read the data... done"
    print "**********************************************************************"

    print "***********************************************************************"
    print "Task 1: Gibbs Sampling"
    print "***********************************************************************"

    # fix K = 20, alpha = 50/K, beta = 0.1, num_iters = 500 for this project
    K = 20
    alpha = 50.0 / K
    beta = 0.1
    num_iters = 500

    print "Run Collapsed Gibbs sampler for LDA..."
    z_indices, C_d, C_t, vocabulary = GibbsSampler(K, alpha, beta, newsgroups_data, num_iters)
    print "Write 5 most frequent words of each topic into a CSV file..."
    WriteFrequentWords(K, C_t, vocabulary)
    print "Task 1 is done..."

    # ##############################################################################
    # # Save the results from Task 1
    # ##############################################################################
    # sio.savemat('z_indices.mat',{'z_indices':z_indices})
    # sio.savemat('C_d.mat', {'C_d':C_d})
    # sio.savemat('C_t.mat', {'C_t':C_t})
    # sio.savemat('vocabulary.mat',{'vocabulary':vocabulary})

    ##############################################################################
    # load the results from Task 1
    ##############################################################################
    # z_indices = sio.loadmat('z_indices.mat',squeeze_me=True)['z_indices']
    # C_d = sio.loadmat('C_d.mat',squeeze_me=True)['C_d']
    # C_t = sio.loadmat('C_t.mat',squeeze_me=True)['C_t']
    # vocabulary = sio.loadmat('vocabulary.mat',squeeze_me=True)['vocabulary']
    

    print "**********************************************************************"
    print "Task 2: Classification                                                "
    print "**********************************************************************"

    # Get topic representations
    topic_rep = getTopicRepresentation(K, alpha, C_d, newsgroups_data)
    # Get bag of words representations
    bag_of_words_rep = getBagOfWords(newsgroups_data, vocabulary)

    # ##############################################################################
    # # save the representation results
    # ##############################################################################
    # sio.savemat('topic_rep.mat',{'topic_rep':topic_rep})
    # sio.savemat('bag_of_words_rep.mat', {'bag_of_words_rep':bag_of_words_rep})


    # ##############################################################################
    # # load the representation results
    # ##############################################################################

    # # topic_rep = sio.loadmat('topic_rep.mat',squeeze_me=True)['topic_rep']
    # # bag_of_words_rep = sio.loadmat('bag_of_words_rep.mat',squeeze_me=True)['bag_of_words_rep']

    learnCurve_mean1, learnCurve_std1, trainingSizes1 = getLearningCurve(topic_rep, newsgroups_labels, 1)
    learnCurve_mean2, learnCurve_std2, trainingSizes2 = getLearningCurve(bag_of_words_rep, newsgroups_labels, 2)


    ################################################################################
    # This part will generate a task2.png image for Task2. Uncomment the following 
    # part is there are any errors related to plotting in the server. 
    ################################################################################
    plt.errorbar(trainingSizes1,learnCurve_mean1, yerr=learnCurve_std1, ecolor='r', fmt='--o',label = "Topic representation")
    plt.errorbar(trainingSizes2,learnCurve_mean2, yerr=learnCurve_std2, ecolor='g', fmt='*', label = "Bag of words representation")
    plt.xlabel("Training sizes")
    plt.ylabel("Error Rate")
    plt.ylim([0, 1.0])
    plt.grid("on")
    plt.title("Compare two representations")
    plt.legend(loc="best")
    plt.savefig('task2.png')













