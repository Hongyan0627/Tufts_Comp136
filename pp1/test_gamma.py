#Build a gamma function for future use
gamma_function_list = []
gamma_function_list.append(1)
gamma_function_list.append(1)
temp = 1
i = 2
N = 64000
K = 10000
while(i < (10.0 * K + N)):
	gamma_function_list.append(i * temp)
	temp = i * temp
	i += 1

print len(gamma_function_list)