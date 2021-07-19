import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import  Dense,Dropout,Flatten
from sklearn.metrics import r2_score
tf.random.set_seed(3)


class reg(object):
    global model
    global size_input
    global size_output
    global thres1
    global negative_positions
    global residula_vector
    global encoder_multiplier_matrix
    global ideal_postions
    global size_1
    global size_2
    global reshape_shape_1
    global reshape_shape_2
    global position
    global order_larger
    global i
    global ls1
    global ls2
    i = 0
    tf.random.set_seed(3)
    
    def model(size_input , size_output):  
      global model
      global i
            #some other activation fuction to try if model improves
        
      if(i==0):
        def my_leaky_relu(x):
          return tf.nn.leaky_relu(x, alpha=0.01)
        layer1 = 2*size_input
        optimizer = tf.keras.optimizers.Adam(learning_rate= 0.01)
        #optimizer = tf.keras.optimizers.SGD()
        model1 = Sequential()
        model1.add(Dense( layer1 ,activation = "relu", input_shape=(size_input,)))
        #model1.add(Dense(layer1,activation = "relu"))
        model1.add(Dense(size_output ))
        model1.compile(optimizer = optimizer  ,loss= 'mse',metrics=['mae', 'mse'])
        model1.summary()
        i = i+1
        model = model1
    
    def get_order_parameters_and_indices(t1,t2,thres1):
        # Get shapes, will help later in unravelling
        global shape1 ,shape2 , size1 , size2 , thres
        
        thres = thres1
        s1=t1.shape;s2=t2.shape
        
        #Flatten the array for further processing, stuff works better with linear arrays
        t1=t1.flatten();t2=t2.flatten()
    
        #Get the number of order parameters in each amplitude matrix
        l1=len(t1[np.abs(t1)>thres]);l2=len(t2[np.abs(t2)>thres])

    
        #If the number is zero, no contribution from the arrays, else get the exact indices, and value of order parameters
        if l1==0:
            ls1=[0];op1=[]
        else:
            op1,ls1 = reg.largest_indices(t1,l1,s1)
        if l2==0:
            ls2=[0];op2=[]
        else:
            op2,ls2= reg.largest_indices(t2,l2,s2)
        #Full order parameters are to be constructed
        order_params=np.concatenate((op1,op2))
    
        return order_params,ls1,ls2
    
    
    
    def reg_train(t1 , t2 , thres):
        
        import time
        global thres1
        global size_input
        global size_output
        global negative_positions
        global residula_vector
        global encoder_multiplier_matrix
        global ideal_postions
        global size_1
        global size_2
        global reshape_shape_1
        global reshape_shape_2
        global position
        global order_larger
        global i 
        global ls1
        global ls2
        
        thres1 = thres
        train = t2
        train_1 = t1
        size_1 = train.size
        size_2 = train_1.size
        reshape_shape_1 = train.shape
        reshape_shape_2 = train_1.shape
        
        train = train.flatten()
        train_1 = train_1.flatten()
        train = np.concatenate((train , train_1) , axis = 0)
         
        #converting negative into positives
        if(i==0):
          negative_positions = train>0
          negative_positions = (negative_positions*2  - 1 )
          train = np.abs(train)
          position = np.abs(train)>thres
          order_larger = np.abs(train)<thres
          order = train[np.abs(train)>thres]
          size_output = train.size
          size_input = order.size
        #absolute_value
        train = np.abs(train)
        
        
        #prediction data
        
        #reshaping for model as [1,size]
       
        
        train = np.reshape(train, (1, size_output))
        
        
        
        #encoder
        order_2 = train>0.01
        data = train*order_2*100
        order_3 = (train>0.001) - 1* order_2
        data = data + train*order_3*1000
        order_4 = (train>0.0001) - 1*order_2 - 1*order_3
        data = data + train*order_4*10000
        order_5 = (train>0.00001 )- 1*order_2 - 1*order_4 - 1*order_3
        data = data + train*order_5*100000
        order_6 = (train>0.000001) - 1*order_2 - 1*order_5 - 1*order_4 - 1*order_3
        data = data + train*order_6*1000000
        order_7 = (train>0.0000001)- 1*order_2 - 1*order_6 - 1*order_5 - 1*order_4 - 1*order_3
        data = data + train*order_7*10000000
        order_8 = (train>0.00000001)- 1*order_2 - 1*order_7 - 1*order_6 - 1*order_5 - 1*order_4 - 1*order_3
        data = data + train*order_8*100000000
        order_9 = (train>(10**-9))- 1*order_2 - 1*order_7 - 1*order_6 - 1*order_5 - 1*order_4 - 1*order_3 - 1*order_8
        data = data + train*order_9*10**9
        order_10 = (train>(10**-10))- 1*order_2 - 1*order_7 - 1*order_6 - 1*order_5 - 1*order_4 - 1*order_3 - 1*order_8 - 1*order_9
        data = data + train*order_10*10**10
        order_11 = (train>(10**-11))- 1*order_2 - 1*order_7 - 1*order_6 - 1*order_5 - 1*order_4 - 1*order_3 - 1*order_8 - 1*order_9 - 1*order_10
        data = data + train*order_11*10**11
        order_12 = (train>(10**-12))- 1*order_2 - 1*order_7 - 1*order_6 - 1*order_5 - 1*order_4 - 1*order_3 - 1*order_8 - 1*order_9 - 1*order_10 - 1*order_11
        data = data + train*order_12*10**12
        order_13 = (train>(10**-13))- 1*order_2 - 1*order_7 - 1*order_6 - 1*order_5 - 1*order_4 - 1*order_3 - 1*order_8 - 1*order_9 - 1*order_10 - 1*order_11 - 1*order_12
        data = data + train*order_13*10**13
        order_14 = (train>(10**-14))- 1*order_2 - 1*order_7 - 1*order_6 - 1*order_5 - 1*order_4 - 1*order_3 - 1*order_8 - 1*order_9 - 1*order_10 - 1*order_11 - 1*order_12 - 1*order_13
        data = data + train*order_14*10**14
        order_15 = (train>(10**-15))- 1*order_2 - 1*order_7 - 1*order_6 - 1*order_5 - 1*order_4 - 1*order_3 - 1*order_8 - 1*order_9 - 1*order_10 - 1*order_11 - 1*order_12 - 1*order_13 - 1*order_14
        data = data + train*order_15*10**15
        
        encoder_multiplier_matrix = .01 * order_2 + 0.001 * order_3 + 0.0001 *order_4 + 0.00001 *order_5 + 0.000001 *order_6 + 0.0000001 *order_7 + 0.00000001 *order_8 + (10**-9) * order_9  + (10**-10) * order_10 + (10**-11) * order_11 + (10**-12) * order_12 + (10**-13) * order_13 + (10**-14) * order_14 + (10**-15) * order_15
        print(data.size)

        if( i == 0):
          order , ls1 , ls2 =  reg.get_order_parameters_and_indices(t1,t2,thres)
          order = np.abs(order)
          order = np.reshape(order , (1,order.size))*(1/thres)
          np.save('ls2',ls1)
          np.save('ls4',ls2)
        else:
          order = train*position
          order = order[order>0]
          order = np.abs(order)
          order = np.reshape(order , (1,order.size))*(1/thres)
        
        print(order.size)
        
        #seperating numbers with less than 10^-5
        ideal_postions = train>(10**(-15))
        input_vector = train*ideal_postions
        residula_vector = train - input_vector
        if(i==0):
          model(size_input, size_output)
        
        start = time.time()
        history = model.fit(order,data, epochs=120)
        end = time.time()
        print(end - start)
        
        return ls1 , ls2
        
    def largest_indices(arr,n,s):
        #Get the highest absolutes values 
        indices=np.argpartition(abs(arr),-n)[-n:]
        
        #Arrange them in decreasing format, can be bypassed
        indices=indices[np.argsort(-abs(arr[indices]))]
    
        #Return order parameter values, and their indices in proper form
        return arr[indices],np.transpose(np.array(np.unravel_index(indices,s)))
    def reg_predict(X):        
        global size_1
        global size_2
        global reshape_shape_1
        global reshape_shape_2
        global model
        global negative_positions
        X = X.flatten()
        X = (np.abs(X))*(1/thres)
        X =np.reshape(X,(1,X.size))
        output = model.predict(X)*encoder_multiplier_matrix
        output = output*ideal_postions + residula_vector
        output = output*negative_positions
        output_1 = np.reshape(output[:,0:size_1], (reshape_shape_1))
        output_2 = np.reshape(output[: , size_1:(size_1+size_2)], (reshape_shape_2))
        return output_1 , output_2
        
    
    def accu(X,Y):
        #calculating accuracy of original and predicted tensor, run only after reg_predict
        from sklearn.metrics import r2_score
        size = X.size
        x1 = np.reshape(X, (size))
        y1 = np.reshape(Y, (size))
        print(r2_score(x1, y1))
        return (r2_score(x1, y1))
        
