################################################################################
#  Import necessary packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import warnings
import time

# Turn off warnings
warnings.filterwarnings("ignore")

################################################################################
# Read in datasets
################################################################################

data_files = ["A.csv", "B.csv", "usps.csv"]
label_files = ["labels-A.csv", "labels-B.csv", "labels-usps.csv"]

def getData(data_file, label_file):
    tmp_data = []
    tmp_label = []
    f = open(data_file)
    for row in f:
        tmp_string = row.strip().split(',')
        tmp_float = [float(num) for num in tmp_string]
        tmp_data.append(np.array(tmp_float))
    f.close()
    f = open(label_file)
    for row in f:
        tmp_label.append(float(row.strip()))
    f.close()
    return np.array(tmp_data), np.array(tmp_label)


def getAllData(data_files, label_files):
    num_dataset = len(data_files)
    
    datasets = []
    labels = []

    for i in range(num_dataset):
        tmp = getData(data_files[i], label_files[i])
        datasets.append(tmp[0])
        labels.append(tmp[1])

    datasets = np.array(datasets)
    labels = np.array(labels)

    return datasets, labels

datasets, labels = getAllData(data_files, label_files)

################################################################################
# Task I: Generative vs. Discriminative
################################################################################

print "*******************************************************************"
print " Task I: Generative vs. Discriminative                             "
print "*******************************************************************"
print "This part takes about 100 seconds..."

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

def getErrorRate(w0, w, test_data, test_label):
    """
    Input: w0 and w are parameters from training data
    Return: predict accuracy on test dataset. 
    """
    accu = 0.0
    for i in range(len(test_label)):
        predit = sigmoid(np.dot(w, test_data[i]) + w0) > 0.5
        if((predit == True and test_label[i] == 1.0) or (predit == False and test_label[i] == 0.0)):
            accu += 1.0
    return 1.0 - accu / len(test_label)


def generativeModel(train_data, train_label):

    class1_data = train_data[train_label == 1.0]
    class2_data = train_data[train_label == 0.0]

    mu1 = np.mean(class1_data, axis = 0)
    mu2 = np.mean(class2_data, axis = 0)
    
    S1  = np.cov(class1_data.transpose())
    S2  = np.cov(class2_data.transpose())
    
    N = len(train_label) * 1.0
    N1 = len(class1_data) * 1.0
    N2 = len(class2_data) * 1.0
    
    S = (N1 * S1 + N2 * S2) / N
    pai = N1 / (N1 + N2)
    S_inv = np.linalg.inv(S)

    w0 = -0.5 * np.dot(np.dot(mu1, S_inv), mu1) + 0.5 * np.dot(np.dot(mu2, S_inv), mu2) + np.log(pai / (1 - pai))
    w = np.dot(S_inv, mu1 - mu2)
    return w0, w

def discriminativeModel(train_data, train_label):
    
    alpha = 0.1
    
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

def getLearningCurve(data, label, model = 0, ratio = 2.0/3.0, random_times = 30):

    learnCurve = []
    train_sizes = np.arange(0.2,1.2,0.2)

    for dummy in range(random_times):
        train_data, train_label, test_data, test_label = split_train_test(data, label, ratio)
        tmp_perf = []
        for train_size in train_sizes:
            real_train_data, real_train_label, dummy_test_data, dummy_test_label = split_train_test(train_data, train_label, train_size)
            if(model == 0):
                w0, w = generativeModel(real_train_data, real_train_label)
                tmp_perf.append(getErrorRate(w0, w, test_data, test_label))
            else:
                w0, w = discriminativeModel(real_train_data, real_train_label)
                tmp_perf.append(getErrorRate(w0, w, test_data, test_label))

        learnCurve.append(np.array(tmp_perf))
    return np.array(learnCurve), len(label) * ratio * train_sizes

def plotErrorBar(dataset, label, title):
    print "Processing dataset: " + title + "..."
    print "Generative Model..."
    learnCurve_g, trainingSizes_g = getLearningCurve(dataset, label, 0)
    print learnCurve_g
    print "Discriminative Model..."
    learnCurve_d, trainingSizes_d = getLearningCurve(dataset, label, 1)
    print learnCurve_d
    ############################################################################
    # Plotting for Task 1 is commented. Uncomment following lines if you want plot
    ############################################################################
    # plt.errorbar(trainingSizes_g,np.mean(learnCurve_g, axis = 0),yerr=np.std(learnCurve_g, axis = 0),ecolor='r', label = "Generative Model")
    # plt.errorbar(trainingSizes_d,np.mean(learnCurve_d, axis = 0),yerr=np.std(learnCurve_d, axis = 0),ecolor='g', label = "Discriminative Model")
    # plt.xlabel("Training sizes")
    # plt.ylabel("Error Rate")
    # plt.grid("on")
    # plt.title("dataset: " + title)
    # plt.legend(loc="best")
    # print "Save image " + title+".png" + " for dataset: " + title + "..."
    # plt.savefig(title + '.png')
    # ##########################################################################


def testBLS():
    print "Test BLS on dataset irls: "
    dataset, label = getData("irlstest.csv", "labels-irlstest.csv")
    true_w = []
    f = open("irlsw.csv")
    for row in f:
        true_w.append(float(row.strip()))
    true_w = np.array(true_w)
    print "Split dataset into training data and test data: "
    train_data, train_label, test_data, test_label = split_train_test(dataset, label, 2.0/3.0)
    print "Apply Generative Model: "
    w0_g, w_g = generativeModel(train_data, train_label)
    print "Apply Discriminative Model: "
    w0_d, w_d = discriminativeModel(train_data, train_label)
    print "True w0: ", true_w[0]
    print "w0 from generative: ", w0_g
    print "w0 from discriminative: ", w0_d
    print "True w: ", true_w[1:]
    print "w from generative: ", w_g
    print "w from discriminative: ", w_d


plotErrorBar(datasets[0], labels[0], 'A')
plotErrorBar(datasets[1], labels[1], 'B')
plotErrorBar(datasets[2], labels[2], 'USPS')

print "Task I: Generative vs. Discriminative ....... is done."
print 
################################################################################
# Task 2: Newton's method vs. gradient ascent
################################################################################

print "*******************************************************************"
print " Task 2: Newton's method vs. gradient descent                      "
print "*******************************************************************"
print "This part takes about 300 seconds..."
def NewtonMethod(train_data, train_label, alpha = 0.1):
    runtimes = []
    w_vals = []

    print "Processing Newton Method..."

    for dummy in range(3):
        print "repeat " + str(dummy + 1) + " times..."
        tmp_runtimes = [] 

        new_train_data = np.ones((len(train_data), 1), dtype = float)
        new_train_data = np.append(new_train_data, train_data, axis = 1)

        num_features = len(new_train_data[0])

        start_time = time.clock()

        w_prev = np.zeros(num_features)
        
        y = np.array([sigmoid(np.dot(w_prev, new_train_data[i])) for i in range(len(new_train_data))])

        R = np.diag(np.multiply(y, 1- y))
        

        part1 = np.linalg.inv(alpha * np.identity(num_features) + np.dot(np.dot(new_train_data.transpose(), R), new_train_data))
        part2 = np.dot(new_train_data.transpose(), (y - train_label)) + alpha * w_prev

        w_next = w_prev - np.dot(part1, part2)

        count = 0

        end_time = time.clock()
        tmp_runtimes.append(end_time - start_time)
        if(dummy == 0):
            w_vals.append(np.array(w_next))

        while(count == 0 or (count < 100 and (np.linalg.norm(w_next - w_prev)/np.linalg.norm(w_prev) >= 0.001))):

            w_prev = w_next

            y = np.array([sigmoid(np.dot(w_prev, new_train_data[i])) for i in range(len(new_train_data))])
            R = np.diag(np.multiply(y, 1- y))
            
            part1 = np.linalg.inv(alpha * np.identity(num_features) + np.dot(np.dot(new_train_data.transpose(), R), new_train_data))
            part2 = np.dot(new_train_data.transpose(), (y - train_label)) + alpha * w_prev
            
            w_next = w_prev - np.dot(part1, part2)
            count += 1
            
            end_time = time.clock()

            tmp_runtimes.append(end_time - start_time)
            if(dummy == 0):
                w_vals.append(np.array(w_next))
        
        runtimes.append(tmp_runtimes)

    return list(np.mean(runtimes, axis = 0)), w_vals

def GradientDescent(train_data, train_label, eta = 0.001, alpha = 0.1):
    
    runtimes = []
    w_vals = []

    print "Processing Gradient Descent..."
    for dummy in range(3):
        if(dummy == 0):
            print "repeat " + str(dummy + 1) + " time..."
        else:
            print "repeat " + str(dummy + 1) + " times..."
        tmp_runtimes = [] 

        new_train_data = np.ones((len(train_data), 1), dtype = float)
        new_train_data = np.append(new_train_data, train_data, axis = 1)

        num_features = len(new_train_data[0])

        start_time = time.clock()

        w_prev = np.zeros(num_features)
        
        y = np.array([sigmoid(np.dot(w_prev, new_train_data[i])) for i in range(len(new_train_data))])

        w_next = w_prev - eta * (np.dot(new_train_data.transpose(), (y - train_label)) + alpha * w_prev)

        end_time = time.clock()
        tmp_runtimes.append(end_time - start_time)
        
        if(dummy == 0):
            w_vals.append(np.array(w_next))

        count = 0

        while(count == 0 or (count < 6000 and (np.linalg.norm(w_next - w_prev)/np.linalg.norm(w_prev) >= 0.001))):

            w_prev = w_next

            y = np.array([sigmoid(np.dot(w_prev, new_train_data[i])) for i in range(len(new_train_data))])
            
            w_next = w_prev - eta * (np.dot(new_train_data.transpose(), (y - train_label)) + alpha * w_prev)
            
            count += 1

            if(count % 60 == 0):
                end_time = time.clock()

                tmp_runtimes.append(end_time - start_time)
                if(dummy == 0):
                    w_vals.append(np.array(w_next))

        runtimes.append(tmp_runtimes)

    return list(np.mean(runtimes, axis = 0)), w_vals


def CompareTwoMethods(dataset, label, title):
    num_data = len(dataset)
    train_data, train_label, test_data, test_label = dataset[num_data/3:], label[num_data/3:], dataset[0:num_data/3], label[0:num_data/3]

    runtimesNM, w_vals_NM = NewtonMethod(train_data, train_label)
    runtimesGD, w_vals_GD = GradientDescent(train_data, train_label)

    print "Calculate Newton Method's performance..."
    error_rates_NM = [getErrorRate(w_vals_NM[i][0], w_vals_NM[i][1:], test_data, test_label) for i in range(len(w_vals_NM))]
    print error_rates_NM
    print "Calculate Gradient Descent's performance..."
    error_rates_GD = [getErrorRate(w_vals_GD[i][0], w_vals_GD[i][1:], test_data, test_label) for i in range(len(w_vals_GD))]
    print error_rates_GD

    tmp_val = error_rates_NM[-1]
    tmp_l1 = len(error_rates_NM)
    tmp_l2 = len(error_rates_GD)
    for i in range(tmp_l1, tmp_l2):
        runtimesNM.append(runtimesGD[i])
        error_rates_NM.append(tmp_val)

    ############################################################################
    # Plotting for Task 2 is commented. Uncomment following lines if you want plot
    ############################################################################
    # plt.plot(runtimesNM, error_rates_NM, 'r*-', label = "Newton Method")
    # plt.plot(runtimesGD, error_rates_GD, 'g*-', label = "Gradient Descent")
    # plt.legend(loc = "best")
    # plt.xlabel("Runtime")
    # plt.ylabel("Error rate")
    # plt.title("dataset: " + title)
    # plt.grid("on")
    # plt.savefig("task2"+title+".png")
    ############################################################################


print "Processing dataset A: ..."
CompareTwoMethods(datasets[0], labels[0], "A")
print "Processing dataset USPS: ..."
CompareTwoMethods(datasets[2], labels[2], "USPS")


print "Task II: Newton's method vs. Gradient Descent ....... is done."
print 
################################################################################
# Task 3: Explore Gradient Desent's Learning Rate
################################################################################

print "*******************************************************************"
print " Task 3: Explore Gradient Desent's Learning Rate                   "
print "*******************************************************************"

def GradientDesentDE(train_data, train_label, test_data, test_label):
    print r"Compare different fixed $\eta$"
    eta_vals = [0.1]

    for eta in eta_vals:
        runtimes, w_vals = GradientDescent(train_data, train_label, eta)
        error_rates = [getErrorRate(w_vals[i][0], w_vals[i][1:], test_data, test_label) for i in range(len(w_vals))]
        #plt.plot(runtimes, error_rates, 'y^-',label = r"$\eta = $" + str(eta))
    # plt.xlabel("Runtimes")
    # plt.ylabel("Error Rate")
    # plt.grid("on")
    # plt.title("Dataset A, learning rate: "+str(eta))
    # #plt.legend(loc="best")
    # plt.savefig("task3_fig3.png")


def LossFunction(w, y, train_data, train_label, alpha = 0.1):
    error = -1.0 * np.sum(np.dot(train_label, np.log(y)) + np.dot(1.0 - train_label, np.log(1.0 - y))) + 0.5 * alpha * np.dot(w,w) 
    return error

def LostFunctionPrime(w, y, train_data, train_label, alpha = 0.1):
    res = np.dot(train_data.transpose(), (y - train_label)) + alpha * w
    return res

def TestStepCond(w, lossPrime, eta, train_data, train_label, alpha = 0.1):
    y1 = []
    y2 = []
    w1 = w - eta * lossPrime
    w2 = w
    
    for i in range(len(train_data)):
        y1.append(sigmoid(np.dot(w1, train_data[i])))
        y2.append(sigmoid(np.dot(w2, train_data[i])))
    y1 = np.array(y1)
    y2 = np.array(y2)

    leftH = LossFunction(w1, y1, train_data, train_label, alpha)
    rightH = LossFunction(w2, y2, train_data, train_label, alpha) - (eta/2.0) * np.sum(np.square(lossPrime))
    
    return leftH > rightH 

def GradientDescentLineSearch(train_data, train_label, alpha = 0.1):
    
    runtimes = []
    w_vals = []
    eta = 0.5
    beta = 0.5

    print "Processing Gradient Descent with Backtrack line Search..."
    for dummy in range(3):
        if(dummy == 0):
            print "repeat " + str(dummy + 1) + " time..."
        else:
            print "repeat " + str(dummy + 1) + " times..."
        tmp_runtimes = [] 

        new_train_data = np.ones((len(train_data), 1), dtype = float)
        new_train_data = np.append(new_train_data, train_data, axis = 1)

        num_features = len(new_train_data[0])

        

        w_prev = np.zeros(num_features)

        start_time = time.clock()
        y = np.array([sigmoid(np.dot(w_prev, new_train_data[i])) for i in range(len(new_train_data))])

        lossPrime = LostFunctionPrime(w_prev, y, new_train_data, train_label)

        tmp_count = 0
        
        while(TestStepCond(w_prev, lossPrime, eta, new_train_data, train_label)):
            tmp_count += 1
            eta = eta * beta
            if(tmp_count > 20):
                break

        w_next = w_prev - eta * lossPrime
        
        eta = 0.5

        end_time = time.clock()

        tmp_runtimes.append(end_time - start_time)
        
        if(dummy == 0):
            w_vals.append(np.array(w_next))

        count = 0

        while(count == 0 or (count < 6000 and (np.linalg.norm(w_next - w_prev)/np.linalg.norm(w_prev) >= 0.001))):
            
            w_prev = w_next

            y = np.array([sigmoid(np.dot(w_prev, new_train_data[i])) for i in range(len(new_train_data))])
            lossPrime = LostFunctionPrime(w_prev, y, new_train_data, train_label)

            tmp_count = 0
            while(TestStepCond(w_prev, lossPrime, eta, new_train_data, train_label, alpha)):
                tmp_count += 1
                eta = eta * beta
                if(tmp_count > 20):
                    break

            w_next = w_prev - eta * lossPrime
            
            eta = 0.5

            count += 1

            if(count % 60 == 0):

                end_time = time.clock()

                tmp_runtimes.append(end_time - start_time)
                if(dummy == 0):
                    w_vals.append(np.array(w_next))

        runtimes.append(tmp_runtimes)

    return list(np.mean(runtimes, axis = 0)), w_vals

def PlotGradientDescent(train_data, train_label, test_data, test_label,alpha = 0.1):
    runtimes, w_vals = GradientDescentLineSearch(train_data, train_label)
    error_rates = [getErrorRate(w_vals[i][0], w_vals[i][1:], test_data, test_label) for i in range(len(w_vals))]
    print error_rates
    
    # plt.plot(runtimes, error_rates, 'r*-')
    # plt.xlabel("Runtimes")
    # plt.ylabel("Error Rate")
    # plt.grid("on")
    # plt.title("Dataset A, Backtrack Line Search")
    # plt.savefig("task3_linesearch.png")

print "Running Gradient Descent with Backtrack line search on dataset A..."
print "This part takes about 198 seconds ..."
data_num = 0
PlotGradientDescent(datasets[data_num][len(datasets[data_num])/3:], labels[data_num][len(datasets[data_num])/3:], datasets[data_num][0:len(datasets[data_num])/3], labels[data_num][0:len(datasets[data_num])/3])
print "Task III: Explore Gradient Desent's Learning Rate ....... is done."
print 


