######################################################################
#Codes for Programming Project 1, Fall 2016 COMP 136
#Author: Hongyan Wang
#Date: 09/18/2016
######################################################################


######################################################################
#Import necessary packages
######################################################################
import numpy as np
import matplotlib.pyplot as plt


######################################################################
# Read the train data and test data from txt files
# Build the full dictionary from training and test data
# ####################################################################
training_data_file = open("./training_data.txt")
test_data_file = open("./test_data.txt")

training_data = []
test_data = []

count = 0
for row in training_data_file:
	training_data = row.strip().split(' ')
training_data_file.close()

for row in test_data_file:
	test_data = row.strip().split(' ')
test_data_file.close()

# number of words in the entire training data
N = len(training_data)

# build full dictionary from train and test data
full_dic = {}
for word in training_data:
	full_dic[word] = full_dic.get(word,0) + 1

for word in test_data:
	full_dic[word] = full_dic.get(word,0) + 1

# number of distinct words
distinct_words = full_dic.keys()
K = len(distinct_words)

#######################################################################
# Task 1: Model Training, Prediction and Evaluation
#######################################################################
# set parameter manually
alpha_prime = 2
alpha_zero = alpha_prime * 1.0 * K

#train size
training_sizes = [N/128, N/64, N/16, N/4, N]

#perplexity for different methods
PP_ML_train = []
PP_MAP_train = []
PP_pd_train =[]

PP_ML_test = []
PP_MAP_test = []
PP_pd_test =[]

#loop over different training size
for i in range(len(training_sizes)):
	training_size = training_sizes[i]
	
	#words frequency in training data
	training_dic = {}
	for j in range(training_size):
		training_dic[training_data[j]] = training_dic.get(training_data[j],0.0) + 1.0
	
	#build three models
	p_ML = {}
	p_MAP = {}
	p_pd = {}
	for k in range(K):
		# if m_k = 0, set m_k = 1
		m_k = training_dic.get(distinct_words[k],0.01)
		alpha_k = alpha_prime
		p_ML[distinct_words[k]] = m_k * 1.0 / training_size
		p_MAP[distinct_words[k]] = (m_k + alpha_k - 1) * 1.0 / (training_size + alpha_zero - K)
		p_pd[distinct_words[k]] = (m_k + alpha_k) * 1.0 / (training_size + alpha_zero)

	# compute the perplexity on full train data
	temp_PP_ML_train = 0.0
	temp_PP_MAP_train = 0.0
	temp_PP_pd_train = 0.0

	for j in range(training_size):
		temp_PP_ML_train += np.log(p_ML[training_data[j]])
		temp_PP_MAP_train += np.log(p_MAP[training_data[j]])
		temp_PP_pd_train += np.log(p_pd[training_data[j]])
	
	temp_PP_ML_train = np.exp(temp_PP_ML_train * (-1.0) / training_size)
	temp_PP_MAP_train = np.exp(temp_PP_MAP_train * (-1.0) / training_size)
	temp_PP_pd_train = np.exp(temp_PP_pd_train * (-1.0) / training_size)

	PP_ML_train.append(temp_PP_ML_train)
	PP_MAP_train.append(temp_PP_MAP_train)
	PP_pd_train.append(temp_PP_pd_train)

	# compute the perplexity on test data
	temp_PP_ML_test = 0.0
	temp_PP_MAP_test = 0.0
	temp_PP_pd_test = 0.0

	for j in range(len(test_data)):
		temp_PP_ML_test += np.log(p_ML[test_data[j]])
		temp_PP_MAP_test += np.log(p_MAP[test_data[j]])
		temp_PP_pd_test += np.log(p_pd[test_data[j]])

	temp_PP_ML_test = np.exp(temp_PP_ML_test * (-1.0) / len(test_data))
	temp_PP_MAP_test = np.exp(temp_PP_MAP_test * (-1.0) / len(test_data))
	temp_PP_pd_test = np.exp(temp_PP_pd_test * (-1.0) / len(test_data))

	PP_ML_test.append(temp_PP_ML_test)
	PP_MAP_test.append(temp_PP_MAP_test)
	PP_pd_test.append(temp_PP_pd_test)

##############################################################################
# print out the result
##############################################################################
print "***************************************************************"
print "Task I: Model Training, Prediction and Evaluation              "
print "***************************************************************"
print "ML estimate with training data sizes N/128, N/64, N/16, N/4, N:"
print "***************************************************************"
print "Perplexities on training data"
print PP_ML_train
print "Perplexities on test data"
print PP_ML_test
print "****************************************************************"
print "MAP estimate with training data sizes N/128, N/64, N/16, N/4, N:"
print "****************************************************************"
print "Perplexities on training data: "
print PP_MAP_train
print "Perplexities on test data: "
print PP_MAP_test
print "************************************************************************************"
print "Predictive distribution estimate with training data sizes N/128, N/64, N/16, N/4, N:"
print "************************************************************************************"
print "Perplexities on training data"
print PP_pd_train
print "Perplexities on test data"
print PP_pd_test

# ###############################################################################
# #Plotting the results for task 1
# ###############################################################################
# plt.plot(training_sizes, PP_ML_train,'rv--',label='ML_train')
# plt.plot(training_sizes, PP_ML_test,'r^--' ,label='ML_test')
# plt.plot(training_sizes, PP_MAP_train,'bD-',label='MAP_train')
# plt.plot(training_sizes, PP_MAP_test,'bx--',label='MAP_test')
# plt.plot(training_sizes, PP_pd_train,'g*-' ,label='PD_train')
# plt.plot(training_sizes, PP_pd_test,'g8--' ,label='PD_test')
# plt.xlabel('training data size')
# plt.ylabel('Perplexities')
# plt.title('Figure 1: The Perplexities on the train and test data under three models')
# plt.legend()
# plt.grid()
# plt.show()


##############################################################################
# Task 2: Model Selection
##############################################################################

#set alpha parameter range
alpha_prime_list = range(1,11,1)

#fixed training size
training_size = N/128

PP_pd_test = []
log_evidence = []

for alpha_prime in alpha_prime_list:

	alpha_zero = K * alpha_prime
	
	# compute the log evidence on training data
	temp_log_evidence = 0.0
	for i in range(training_size):
		temp_log_evidence += (-1.0) * np.log(alpha_zero + i)

	training_dic = {}
	for j in range(training_size):
		training_dic[training_data[j]] = training_dic.get(training_data[j],0) + 1
	
	# compute the perplexity on test data and log evidence on training data
	p_pd = {}

	for k in range(K):
		# if m_k = 0, set m_k = 0.01
		m_k = training_dic.get(distinct_words[k],0.01)
		alpha_k = alpha_prime
		p_pd[distinct_words[k]] = (m_k + alpha_k) * 1.0 / (training_size + alpha_zero)

		if(m_k >= 1):
			for i in range(m_k):
				temp_log_evidence += np.log(alpha_k + i)

	temp_PP_pd_test = 0.0
	for j in range(len(test_data)):
		temp_PP_pd_test += np.log(p_pd[test_data[j]])
	temp_PP_pd_test = np.exp(temp_PP_pd_test * (-1.0) / len(test_data))

	PP_pd_test.append(temp_PP_pd_test)
	log_evidence.append(temp_log_evidence)

PP_pd_test = [(int)(item) for item in PP_pd_test]
log_evidence = [(int)(item) for item in log_evidence]

###############################################################################
##Print out the results for task 2
###############################################################################
print "***************************************************************"
print "Task II: Model Selection                                       "
print "***************************************************************"
print "The Perplexities on test set for alpha_prime = 1.0, ...., 10.0 "
print PP_pd_test
print "The log evidence for alpha_prime = 1.0, ... , 10.0"
print log_evidence

# ##############################################################################
# #Plotting the results for task 2
# ##############################################################################
# plt.figure(1)
# plt.subplot(121)
# plt.plot(alpha_prime_list, PP_pd_test,'r*-')
# plt.xlabel('alpha prime')
# plt.ylabel('Perplexities on test data')
# plt.title('Figure 2: The Perplexities on the test data for predictive distribution model')
# plt.grid()

# plt.subplot(122)
# plt.plot(alpha_prime_list, log_evidence,'b*-')
# plt.xlabel('alpha prime')
# plt.ylabel('log evidence on training data')
# plt.title('Figure 3: The log evidence on the training data')
# plt.grid()
# plt.show()


##############################################################################
# Task 3: Author Identification
##############################################################################

full_dic = {}
training_dic = {}

test_data84 = []
test_data1188 = []
train_data345 = []

# read in the training data
f = open("./pg345.txt.clean.txt")
for row in f:
	temp_words = row.strip().split(' ')
	for word in temp_words:
		if(len(word.strip()) < 1):
			continue
		full_dic[word] = full_dic.get(word,0) + 1
		train_data345.append(word)
		training_dic[word] = training_dic.get(word,0) + 1
f.close()

# read in the test data
f = open("./pg84.txt.clean.txt")
for row in f:
	temp_words = row.strip().split(' ')
	for word in temp_words:
		if(len(word.strip()) < 1):
			continue
		full_dic[word] = full_dic.get(word,0) + 1
		test_data84.append(word)
f.close()


# read in the test data
f = open("./pg1188.txt.clean.txt")
for row in f:
	temp_words = row.strip().split(' ')
	for word in temp_words:
		if(len(word.strip()) < 1):
			continue
		full_dic[word] = full_dic.get(word,0) + 1
		test_data1188.append(word)
f.close()

total_words = full_dic.keys()
K = len(total_words)
training_size = len(train_data345)

#set parameter
alpha_prime = 2.0
alpha_zero = K * alpha_prime


p_pd = {}
for k in range(K):
	# if m_k = 0, set m_k = 0.01
	m_k = training_dic.get(total_words[k],0.01)
	alpha_k = alpha_prime
	p_pd[total_words[k]] = (m_k + alpha_k) * 1.0 / (training_size + alpha_zero)

# compute perplexity on two test data sets
PP_pd_test84 = 0.0
PP_pd_test1188 = 0.0

for j in range(len(test_data84)):
	PP_pd_test84 += np.log(p_pd[test_data84[j]])
PP_pd_test84 = np.exp(PP_pd_test84 * (-1.0) / len(test_data84))

for j in range(len(test_data1188)):
	PP_pd_test1188 += np.log(p_pd[test_data1188[j]])
PP_pd_test1188 = np.exp(PP_pd_test1188 * (-1.0) / len(test_data1188))


# ###############################################################################
# ##Print out the results for task 3
# ###############################################################################
print "**********************************************************************"
print "Perplexities on pg84.txt.clean.txt"
print PP_pd_test84
print "**********************************************************************"
print "Perplexities on pg1188.txt.clean.txt"
print PP_pd_test1188







