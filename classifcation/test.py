from numpy import genfromtxt
my_data = genfromtxt('data_dir/amazon/y_test.csv', delimiter=',')
print(my_data)