#
# import necessary packages
#
import matplotlib.pyplot as plt
import numpy as np
import random

##########################################################################
# Define helper functions
##########################################################################

datasets_names = ["-100-10","-100-100","-1000-100","-crime","-wine"]
train_test_names = ["./train","./trainR","./test","./testR"]

def read_file(file_name):
    """
    input: file name
    return: n by m, np array, n is number of data, m is number of features  
    """
    myfile = open(file_name)
    result = []
    for row in myfile:
        temp_string_array = row.strip().split(',')
        temp_num_array = [float(item) for item in temp_string_array]
        if(len(temp_num_array) > 1):
            result.append(temp_num_array)
        else:
            result.append(temp_num_array[0])
    result = np.array(result)
    myfile.close()
    return result;

def get_all_data():
    """
    return all the trainning data, training label, test data, test label
    """
    
    all_data = [[] for dummy in range(len(train_test_names))]
    for num1 in range(len(datasets_names)):
        for num2 in range(len(train_test_names)):
            file_name = train_test_names[num2] + datasets_names[num1] + ".csv"
            all_data[num2].append(read_file(file_name))
    
    for num in range(3):
        all_data[0].append(all_data[0][2][0:(50 * (num + 1))])
        all_data[1].append(all_data[1][2][0:(50 * (num + 1))])
        all_data[2].append(all_data[2][2])
        all_data[3].append(all_data[3][2])

    for num in range(len(all_data)):
        all_data[num] = np.array(all_data[num])
    
    return all_data

def compute_MSE(phi_matrix, target, w_parameters):
    """
    input: phi_matrix is the n By m data matrix, n is the number of data, m is 
            the number of features
           target is n By 1 labels, w_parameters is m by 1 parameters
    return: MSE = sum(|phi_matrix * w_parameters - target|.^2)/n
    """
    num_data = phi_matrix.shape[0]
    predict = np.dot(phi_matrix,w_parameters)
    diff = predict - target
    mse = np.sum(np.square(diff)) * 1.0 / num_data
    return mse

def compute_W(phi_matrix, target, lambda_val = 0.0):
    """
    input: phi_matrix is the n By m data matrix, n is the number of data, m is 
            the number of features
           target is n By 1 labels, lambda_val is regularization parameter
    return: w_parameters, which is m by 1 array
    """
    num_features = phi_matrix.shape[1]
    mat = lambda_val * np.identity(num_features) + np.dot(phi_matrix.transpose(),phi_matrix)
    inverse_mat = np.linalg.pinv(mat)
    w_parameters = np.dot(np.dot(inverse_mat,phi_matrix.transpose()),target)
    return w_parameters

##############################################################################
# read in datasets
##############################################################################

print "load in data......"

all_data    = get_all_data()

train_data  = all_data[0]
train_label = all_data[1]
test_data   = all_data[2]
test_label  = all_data[3]

lambda_vals = range(1,151)

datasets_names.append("-50(1000)-100")
datasets_names.append("-100(1000)-100")
datasets_names.append("-150(1000)-100")

print "load in data......done"

##############################################################################
#                    Task 1: Regularization
##############################################################################

print "############################################################"
print "#             Task 1: Regularization                       #"
print "############################################################"

def get_mse(train_data, train_label, test_data, test_label, lambda_vals):
    train_mse = []
    test_mse = []
    for num1 in range(len(train_data)):
        train_tmp = []
        test_tmp = []
        for num2 in range(len(lambda_vals)):
            lambda_val = lambda_vals[num2]
            w_parameters = compute_W(train_data[num1],train_label[num1],lambda_val)
            train_tmp.append(compute_MSE(train_data[num1],train_label[num1],w_parameters))
            test_tmp.append(compute_MSE(test_data[num1],test_label[num1],w_parameters))
        train_mse.append(train_tmp)
        test_mse.append(test_tmp)
    return train_mse,test_mse

train_mse,test_mse = get_mse(train_data,train_label,test_data,test_label,lambda_vals)

task1_optimal_lambda = []
task1_test_mse = []

for num in range(len(train_mse)):
    tmp_min_err = float('inf')
    tmp_min_lambda = 0
    for i in range(len(test_mse[num])):
        if(test_mse[num][i] < tmp_min_err):
            tmp_min_err = test_mse[num][i]
            tmp_min_lambda = lambda_vals[i]
    task1_optimal_lambda.append(tmp_min_lambda)
    task1_test_mse.append(tmp_min_err)

for num in range(len(datasets_names)):
    print "Dataset " + datasets_names[num] + ": optimal lambda is " + str(task1_optimal_lambda[num]) + " and MSE on test set is " + str(task1_test_mse[num])



################################################################################
# plot the results for Task 1
################################################################################

# for num in range(len(train_mse)):
#     plt.figure(num + 1)
#     plt.plot(lambda_vals, train_mse[num],'b',label="train"+datasets_names[num])
#     plt.plot(lambda_vals, test_mse[num],'r^',label="test"+datasets_names[num])
#     if(num == 0 or num == 5):
#         plt.plot(lambda_vals,[3.78 for dummy in range(len(lambda_vals))],'y*',label="true MSE")
#     elif(num == 1 or num == 6):
#         plt.plot(lambda_vals,[3.78 for dummy in range(len(lambda_vals))],'y*',label="true MSE")
#     elif(num == 2 or num == 7):
#         plt.plot(lambda_vals,[4.015 for dummy in range(len(lambda_vals))],'y*',label="true MSE")
#     else:
#         pass
#     plt.title("Figure " + str(num + 1) + ": MSE on dataset " + datasets_names[num])
#     plt.xlabel("lambda")
#     plt.ylabel("MSE")
#     plt.grid("on")
#     plt.legend(loc="best")
#     plt.show()
#     #plt.savefig("task1_figure"+str(num+1)+".png");
################################################################################

print "Task 1 is done....You can uncomment the plotting part to see the plots."



##############################################################################
#                    Task 2: Learning Curves
##############################################################################

print "############################################################"
print "#             Task 2: Learning Curves                      #"
print "############################################################"

task2_lambda_vals = [lambda_vals[0],np.argmin(test_mse[2]),lambda_vals[-1]]
task2_train_data  = train_data[2]
task2_train_label = train_label[2]
task2_test_data   = test_data[2]
task2_test_label  = test_label[2]
task2_training_sizes = range(10,810,10)

def get_learning_curves(train_data1,train_label1,test_data1,test_label1,training_sizes, lambda_vals,random_times = 10):
    learning_curves = [[] for dummy in range(len(lambda_vals))]
    train_data  = train_data1
    train_label = train_label1
    test_data   = test_data1
    test_label  = test_label1
    
    for num in range(len(learning_curves)):
        lambda_val = lambda_vals[num]
        for training_size in training_sizes:
            tmp_mse = 0.0
            for dummy in range(random_times):
                combined_data = zip(train_data,train_label)
                np.random.shuffle(combined_data)
                train_data  = [combined_data[i][0] for i in range(len(combined_data))]
                train_label = [combined_data[i][1] for i in range(len(combined_data))]
                train_data  = np.array(train_data)
                train_label = np.array(train_label)
                tmp_w = compute_W(train_data[0:training_size], train_label[0:training_size],lambda_val)
                tmp_mse += compute_MSE(test_data,test_label,tmp_w)
            tmp_mse = tmp_mse * 1.0 / random_times
            learning_curves[num].append(tmp_mse)
    return learning_curves

learning_curves = get_learning_curves(task2_train_data,task2_train_label,task2_test_data,task2_test_label,task2_training_sizes,task2_lambda_vals,10)

# ################################################################################
# # plot the results for Task 2
# ################################################################################

# for num in range(len(learning_curves)):
#     plt.plot(task2_training_sizes,learning_curves[num],label="lambda = " + str(task2_lambda_vals[num]))

# plt.grid("on")
# plt.xlabel("Training size")
# plt.ylabel("MSE")
# plt.legend(loc="best")
# plt.title("Figure 9: Learning curves for different lambda")
# plt.show()

print "Task 2 is done....You can uncomment the plotting part to see the plots."

##############################################################################
#                    Task 3: Cross validation
##############################################################################

print "############################################################"
print "#             Task 3: Cross validation                     #"
print "############################################################"

def cross_validation(train_data,train_label,test_data,test_label,K = 10):
    optimal_lambda = []
    test_mse = []
    for num in range(len(train_data)):
        task3_training_data = train_data[num]
        task3_training_label = train_label[num]
        task3_test_data = test_data[num]
        task3_test_label = test_label[num]
        N = len(task3_training_data)

        split_train_data = [task3_training_data[((N/K) * i) : ((N/K) * (i + 1))] for i in range(K)]
        split_train_label = [task3_training_label[((N/K) * i) : ((N/K) * (i + 1))] for i in range(K)]

        min_error = float('inf')
        min_lambda = 0

        for lambda_val in lambda_vals:
            
            tmp_error = 0.0
            
            for i in range(K):
                tmp_train_data  = []
                tmp_train_label = []
                tmp_test_data = []
                tmp_test_label = []
                if(i == 0):
                    tmp_train_data = task3_training_data[(N/K):]
                    tmp_train_label = task3_training_label[(N/K):]
                    tmp_test_data = task3_training_data[0:(N/K)]
                    tmp_test_label = task3_training_label[0:(N/K)]
                elif(i == (K - 1)):
                    tmp_train_data = task3_training_data[0:((N/K) * (K - 1))]
                    tmp_train_label = task3_training_label[0:((N/K) * (K - 1))]
                    tmp_test_data = task3_training_data[((N/K) * (K - 1)):]
                    tmp_test_label = task3_training_label[((N/K) * (K - 1)):]
                else:
                    tmp_train_data = task3_training_data[0:((N/K) * i)]
                    tmp_train_label = task3_training_label[0:((N/K) * i)]
                    np.append(tmp_train_data, task3_training_data[((N/K) * (i + 1)):], axis = 0)
                    np.append(tmp_train_label, task3_training_label[((N/K) * (i + 1)):], axis = 0)
                    tmp_test_data = task3_training_data[((N/K) * (i)):((N/K) * (i + 1))]
                    tmp_test_label = task3_training_label[((N/K) * (i)):((N/K) * (i + 1))]

                tmp_w = compute_W(tmp_train_data,tmp_train_label,lambda_val)
                tmp_error += compute_MSE(tmp_test_data,tmp_test_label,tmp_w)

            tmp_error = tmp_error/K

            if(tmp_error < min_error):
                min_error = tmp_error
                min_lambda = lambda_val

        optimal_lambda.append(min_lambda)
        w_p = compute_W(task3_training_data,task3_training_label,min_lambda)
        test_mse.append(compute_MSE(task3_test_data,task3_test_label,w_p))
    return optimal_lambda, test_mse

task3_optimal_lambda, task3_test_mse = cross_validation(train_data,train_label,test_data,test_label)

################################################################################
# print the results for Task 3
################################################################################


for num in range(len(datasets_names)):
    print "Dataset " + datasets_names[num] + ": optimal lambda is " + str(task3_optimal_lambda[num]) + " and MSE on test set is " + str(task3_test_mse[num])

print "Task 3 is done...."

##############################################################################
#                    Task 4: Bayesian Model Selection
##############################################################################

print "############################################################"
print "#             Task 4: Bayesian Model Selection             #"
print "############################################################"

def getGamma(alpha, eigenvals):
    gamma = 0.0
    for i in range(len(eigenvals)):
        gamma += np.real(eigenvals[i]) * 1.0 / (alpha + np.real(eigenvals[i]))
    return gamma

def getBeta(gamma, N, mN, phi_matrix, target):
    beta = 0.0
    for i in range(N):
        beta += ((target[i] - np.dot(mN.transpose(), phi_matrix[i])) ** 2)
    beta = beta / (N - gamma)
    beta = 1.0 / beta
    return beta

def getHyp(init_alpha, init_beta,phi_matrix,target, test_data, test_label):
    num_features = phi_matrix.shape[1]
    num_examples = phi_matrix.shape[0]

    sN = np.linalg.pinv(init_alpha * np.identity(num_features) + init_beta * np.dot(phi_matrix.transpose(), phi_matrix))
    mN = init_beta * np.dot(sN, np.dot(phi_matrix.transpose(), target))
    eigenvals = np.linalg.eig(init_beta * np.dot(phi_matrix.transpose(), phi_matrix))[0]
    gamma = getGamma(init_alpha, eigenvals)


    alpha_new = np.real(gamma / (np.dot(mN.transpose(), mN)))
    beta_new = np.real(getBeta(gamma, num_examples, mN, phi_matrix, target))

    alpha_prev = init_alpha
    beta_prev = init_beta
    count = 0

    while((abs(alpha_new - alpha_prev) >= 0.0000001) or (abs(beta_new - beta_prev) >= 0.0000001)):
        
        alpha_prev = alpha_new
        beta_prev = beta_new

        sN = np.linalg.pinv(alpha_prev * np.identity(num_features) + beta_prev * np.dot(phi_matrix.transpose(), phi_matrix))
        mN = beta_prev * np.dot(sN, np.dot(phi_matrix.transpose(), target))
        eigenvals = np.linalg.eig(beta_prev * np.dot(phi_matrix.transpose(), phi_matrix))[0]
        gamma = getGamma(alpha_prev, eigenvals)


        alpha_new = np.real(gamma / (np.dot(mN.transpose(), mN)))
        beta_new = np.real(getBeta(gamma, num_examples, mN, phi_matrix, target))
        count += 1

    lambda_val = alpha_new / beta_new

    w_p = compute_W(phi_matrix, target, lambda_val)
    test_mse = compute_MSE(test_data, test_label, w_p)

    return lambda_val,test_mse, count


task4_optimal_lambda = []
task4_test_mse = []
task4_num_iter = []

for i in range(len(train_data)):
    tmp = getHyp(0.8, 1.2, train_data[i], train_label[i], test_data[i], test_label[i])
    task4_optimal_lambda.append(tmp[0])
    task4_test_mse.append(tmp[1])
    task4_num_iter.append(tmp[2])


################################################################################
# print the results for Task 4
################################################################################


for num in range(len(datasets_names)):
    print "Dataset " + datasets_names[num] + ": optimal lambda is " + str(task4_optimal_lambda[num]) + " and MSE on test set is " + str(task4_test_mse[num]) + "using " + str(task4_num_iter[num]) + " iterations."

print "Task 4 is done...."








