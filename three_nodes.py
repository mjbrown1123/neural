import numpy as np
import gzip
import pickle
#import Theano
 
rate = 0.0001 
training_num = 3000
sample_size = 50000

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
 
#f.close()
 
#train_set_x, train_set_y = shared_dataset(train_set)
 
train_set_x = train_set[0]
train_set_y = train_set[1]
 
batch_size = sample_size
 
 
 
#sets = train_set[0].tolist()
 
#print(sets[0])
#print(train_set[1][0])
 
data = train_set_x[0 : batch_size]
label = train_set_y[0 : batch_size]
"""data = train_set_x
label = train_set_y"""
 
 
print(len(data))
print(len(label))
 
print(get_bit_array(8))
print(get_bit_array(1))
print(get_bit_array(4))
 
print(len(train_set_x))
print(len(train_set_y))
 
 
# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((785,1000)) - 1
syn1 = 2*np.random.random((1000,100)) - 1
syn2 = 2*np.random.random((100,10)) - 1
 
rows, columns = data.shape
 
arr = []
 
l1List =data.tolist()
 
for iter in range(rows):
    """print([1.])
    print(l1List[iter])
    #print(l1[iter].insert(0, 1.))
      
    #print(l1List[iter].insert(0,1.))
    print([1.] + l1List[iter])"""
    arr.append([1.] + l1List[iter])
    print("Finished ", iter)
 
 
    #print(arr)
 
X = np.array(arr)
 
arr = []
 
testt = label.tolist()
 
for iter in range(len(testt)):
    arr.append(get_bit_array(testt[iter]))
 
y = np.array(arr)
print(y)
 
print(rate)
print("learning started")
 
for j in range(training_num):
 
                # Feed forward through layers 0, 1, and 2
    l0 = X  #4x4
    l1 = nonlin(np.dot(l0,syn0)) #4x4
 
    rows,columns = l1.shape
    arr1 = []
    for iter in range(rows):
        arr1.append(1.)
 
    #print(arr1)
    l1[:,0] = arr1
 
    #print(l1)
 
    """rows, columns = l1.shape
 
    arr = []
 
    l1List =l1.tolist()
 
    for iter in range(rows):
        print([1.])
        print(l1List[iter])
        #print(l1[iter].insert(0, 1.))
      
        #print(l1List[iter].insert(0,1.))
        print([1.] + l1List[iter])
        arr.append([1.] + l1List[iter])
 
 
    #print(arr)
 
    lbtest = np.array(arr)
 
    l1 = lbtest  #l1 = 4x5
 
"""
    l2 = nonlin(np.dot(l1,syn1)) #4x1
 
    rows,columns = l2.shape
    arr2 = []
    for iter in range(rows):
        arr2.append(1.)
 
    #print(arr1)
    l2[:,0] = arr2
 
    l3 = nonlin(np.dot(l2,syn2))
 
    #print(l3)
    #print(y)
 
    # how much did we miss the target value?
    l3_error = y - l3 #4x1
 
    l3_delta = l3_error * nonlin(l3,deriv=True)
 
    l2_error = l3_delta.dot(syn2.T)
 
 
   
    if (j%10) == 0:
        print("Error ",j,": " + str(np.mean(np.abs(l3_error))))
       
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True) # 4x1 * 4x1 = 4x1
 
    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T) # 4x1 dot 1x5 = 4x5
   
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True) #  4x5 * 4x5 = 4x5
 
    syn2 += rate* l2.T.dot(l3_delta)
 
    syn1 += rate* l1.T.dot(l2_delta) # 5x4 dot 4x1 = 5x1
 
    """print(l0.T.dot(l1_delta))
    for iter in range(rows):
        print([1.])
        print(l1List[iter])
        #print(l1[iter].insert(0, 1.))
      
        #print(l1List[iter].insert(0,1.))
        print([1.] + l1List[iter])
        arr.append([1.] + l1List[iter])"""
 
    #l1_delta = np.delete(l0.T.dot(l1_delta), 0, 1)
    syn0 += rate*l0.T.dot(l1_delta) # 4x4  dot 4x5 = 4x5
 
 
res = nonlin(np.dot(X, syn0))
rows, columns = res.shape
 
arr2 = []
for iter in range(rows):
    arr2.append(1.)
 
#print(arr1)
res[:,0] = arr2
"""
resList =res.tolist()
resArr = []
 
for iter in range(rows):
    print([1.])
    print(l1List[iter])
    #print(l1[iter].insert(0, 1.))
   
    #print(l1List[iter].insert(0,1.))
    print([1.] + l1List[iter])
    resArr.append([1.] + resList[iter])
 
 
    #print(arr)"""
 
#restest = np.array(resArr)
 
res = nonlin(np.dot(res, syn1))
rows, columns = res.shape
 
arr2 = []
for iter in range(rows):
    arr2.append(1.)
 
#print(arr1)
res[:,0] = arr2
 
"""
resList =res.tolist()
 
resArr = []
 
for iter in range(rows):
    print([1.])
    print(l1List[iter])
    #print(l1[iter].insert(0, 1.))
   
    #print(l1List[iter].insert(0,1.))
    print([1.] + l1List[iter])
    resArr.append([1.] + resList[iter])
 
 
    #print(arr)"""
 
#restest1 =  np.array(resArr)
 
result = np.around(nonlin(np.dot(res,syn2)))
 
print(result)
print("Rate:", rate)
 
 
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
 
 
 
#print(nonlin(np.dot(restest,syn1)))
 
#print(nonlin(np.dot(nonlin(np.dot(nonlin(np.dot(X,syn0)),syn1)),syn2)))
 
"""print(np.append([1],l1))
 
print(lbtest)"""