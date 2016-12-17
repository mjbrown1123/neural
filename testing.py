import numpy as np
import gzip
import pickle
#import Theano


stringing = "helllooowoeifn"

print(stringing[2:5])
 
rate = 0.0001
training_num = 3000
sample_size = 50000

synapse_file = "WEIGHTS_50000_samples_at_3000_reps_rate_0_0001_w_greatest_val.txt"

"""
combos that work:

1. rate: 0.003, training_num: 1000, sample_size: 1000
2. rate: 0.0001, training_num: 1000, sample_size: 10000, 8410/10000
3. rate: 0.0003, training_num: 1000, sample_size: 10000, 8557/10000
"""



def nonlin(x,deriv=False):
                if(deriv==True):
                    return x*(1-x)
 
                return 1/(1+np.exp(-x))
 
#this takes the digit label and returns a bit array that corresponds to this value
def get_bit_array(x):
 
    array = []
 
    for iter in range(10):
        if(iter == x):
            array.append(1)
 
        else:
            array.append(0)
 
 
    return np.array(array)
 
def get_num(x):
   
    for iter in range(len(x)):
        if(x[iter] == 1.):
            return iter
 
    return -1   
 
def shared_dataset(dataxy):
 
    data_x, data_y = dataxy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
 
    return shared_x, T.cast(shared_y, 'int32')
 
X = np.array([[1,0,0,1],
            [1,0,1,1],
            [1,1,0,1],
            [1,1,1,1]])
               
y = np.array([[0],
                                                [1],
                                                [0],
                                                [1]])
 
np.random.seed(1)
 
#sys.setdefaultencoding('utf-8')
 
f = gzip.open('mnist.pkl.gz', 'rb')
 
train_set, valid_set, test_set = pickle.load(f)
 
test_set_x = test_set[0]
test_set_y = test_set[1]


data = test_set_x
label = test_set_y

rows, columns = data.shape
 
arr = []
 
l1List =data.tolist()
 
for iter in range(rows):
    #print([1.])
    #print(l1List[iter])
    #print(l1[iter].insert(0, 1.))
      
    #print(l1List[iter].insert(0,1.))
    #print([1.] + l1List[iter])
    arr.append([1.] + l1List[iter])
    #print("Finished ", iter)
 
 
    #print(arr)

 
X = np.array(arr)
 
arr = []
 
testt = label.tolist()
 
for iter in range(len(testt)):
    arr.append(get_bit_array(testt[iter]))
 
y = np.array(arr)
print(y)
 
print(rate)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((785,1000)) - 1
syn1 = 2*np.random.random((1000,100)) - 1
syn2 = 2*np.random.random((100,10)) - 1


file = open(synapse_file)

lists = file.readlines()


syn0_rows = 785
syn0_columns = 1000

syn1_rows = 1000
syn1_columns = 100

syn2_rows = 100
syn2_columns = 10

columns = [syn0_columns, syn1_columns, syn2_columns]

#good
print(len(lists[1]))

#good
print(len(lists[4]))

#good
print(len(lists[7]))

arr = []

arr.append(lists[1])
arr.append(lists[4])
arr.append(lists[7])

###

#

#
#
#
#


arr2 = []


#ignore
for iter in range(len(arr)):
















    prevIndex = 0
    currentIndex = 0
    full_arr = []
    row_arr = []
    for i in range(len(arr[iter])):

        #this successfully strips values from the string
        if(arr[iter][i] == ','):
            row_arr.append(float(arr[iter][prevIndex:i]))
            prevIndex = i + 1

        #this needs to figure out when to create a new row_array
        if(len(row_arr) == columns[iter]):
            full_arr.append(row_arr)
            row_arr = []


    arr2.append(full_arr)















syn0 = np.array(arr2[0])
syn1 = np.array(arr2[1])
syn2 = np.array(arr2[2])

#

#



#

##

####
"""


    in this section, make sure to add the zeros!!!!


"""
res = nonlin(np.dot(X, syn0))
rows, columns = res.shape
 
arr2 = []
for iter in range(rows):
    arr2.append(1.)
 
#print(arr1)
res[:,0] = arr2
 
#restest = np.array(resArr)
 
res = nonlin(np.dot(res, syn1))
rows, columns = res.shape
 
arr2 = []
for iter in range(rows):
    arr2.append(1.)
 
#print(arr1)
res[:,0] = arr2
 
 
#restest1 =  np.array(resArr)
resulting = nonlin(np.dot(res,syn2))

rows,columns = resulting.shape
result_list = resulting.tolist()

max_arr = []

for iter in range(rows):

    maximum = 0
    max_index = 0

    arr = [0] * columns

    for i in range(columns):

        if(result_list[iter][i] > maximum):
            max_index = i
            maximum = result_list[iter][i]

    arr[max_index] = 1
    max_arr.append(arr)

result = np.array(max_arr)

#result is the final result from dot multiplying 


tesss = result.tolist()
yyy = y.tolist()
 
correct = 0
total = 0
 
for iter in range(len(tesss)):
    if(tesss[iter] == yyy[iter]):
        correct = correct+ 1
 
    #print(tesss[iter], " ", yyy[iter])
    total = total + 1
 
print("Accuracy: ", correct / total, "  Correct: ", correct, "  Total: ", total)
 
 
