class reg(object):  
    from sklearn.linear_model import Ridge
    import numpy as np
    import inp
    alpha = inp.alpha
    global np
    import time
    global model 
    #model = KernelRidge(alpha = alpha)
    global shape1 ,shape2 , size1 , size2
    global i 
    global Amp1
    i = 0
    
    
    
    
    
    # RidgeRegression for multioutput regression
    def fit_fun(X):
      z = np.sin((2*X+1)*(np.pi/2))*np.exp(-X)
      input = np.concatenate((z , np.log(X),(1/X)**2), axis = 1)
      return input
    
    
    
    
    def get_order_parameters_and_indices(t1,t2,thres):
        # Get shapes, will help later in unravelling
        global shape1 ,shape2 , size1 , size2
        shape1=t1.shape;shape2=t2.shape
        
        #for recontruscting the output tensor
        size1 = t1.size
        size2 = t2.size
        
        #Flatten the array for further processing, stuff works better with linear arrays
        t1=t1.flatten();t2=t2.flatten()
    
        #Get the number of order parameters in each amplitude matrix
        l1=len(t1[abs(t1)>thres]);l2=len(t2[abs(t2)>thres])
        #print('l1,l2',l1,l2)
         
        #If the number is zero, no contribution from the arrays, else get the exact indices, and value of order parameters
        if l1==0:
            ls1=[0];op1=[]
        else:
            #ls1=[0]
            #op1=[]
            op1,ls1 = reg.largest_indices(t1,l1,shape1)
        if l2==0:
            ls2=[0];op2=[]
        else:
            op2,ls2= reg.largest_indices(t2,l2,shape2)
        #Full order parameters are to be constructed
        #order_params=np.concatenate((op1,op2))
        #print('get_order_parameter_and_indices')
        #print(op1,op2)
        #print(ls1)
        #print(ls2)
        #np.save('ls2',ls1)
        #np.save('ls4',ls2)
        return np.concatenate((op1,op2)),np.array(ls1),np.array(ls2)
    
    def largest_indices(arr,n,s):
        #Get the highest absolutes values 
        indices=np.argpartition(abs(arr),-n)[-n:]
        
        #Arrange them in decreasing format, can be bypassed
        indices=indices[np.argsort(-abs(arr[indices]))]
        #print('indices',indices)
        #print('shape',s)
        #Return order parameter values, and their indices in proper form
        return arr[indices],np.transpose(np.array(np.unravel_index(indices,s)))
      
      
      
      
    
    
    
    def reg_train(t1 , t2 , thres, alpha1 = 0.0001):
        import time
        #saving the currently passed data to retrain later
        from sklearn.linear_model import Ridge
        #dam = Ridge(alpha=0.0001)
        global model
        global i
        global shape
        global Amp1
        alpha1 = alpha1
        Amp = np.concatenate((t1.flatten(),t2.flatten()),axis = 0)
        Amp = np.reshape(Amp,(1,Amp.size))
        if i==0:
            order, ls2 , ls4 =  reg.get_order_parameters_and_indices(t1,t2,thres)
            order = np.reshape(order,(1,order.size))
            np.save('ls2',ls2)
            np.save('ls4',ls4)
        if i>0:
            ls2=np.load('ls2.npy')
            ls4=np.load('ls4.npy')
            #print(ls2.size)
            if ls2.size==1:
                op1=[]
            else:
                #print('what are you doing here?')
                op1=t1[ls2[:,0],ls2[:,1]]
            if ls4.size==1:
                op2=[]
            else:
                op2=t2[ls4[:,0],ls4[:,1],ls4[:,2],ls4[:,3]]
            order=np.concatenate((np.array(op1),np.array(op2)))
            order=order.reshape(1,order.size)
            #print(order) 
        #print('order parameters')
        #print(ls2)
        #print(ls2)
        #loading the data from previous iterations and form a dataset so after every 
        #iter model trains on bigger dataset
        #----------------------------------------------------------------------------
        if(i>0):
                try:
                    concatX = np.load("temp_train_data_Order.npy")
                    concatY = np.load("temp_train_data_Amp.npy")
                except Exception:
                                pass
                
        
        #concatenating the previous iter data with current data for weights update in model
        #--------------------------------------------------------------------------------
        if(i>0):
                try:
                    Amp = np.concatenate((concatY, Amp),axis = 0)
                    order = np.concatenate((concatX, order),axis = 0)
                except Exception:
                                pass
        
        #---------------------------------------------------------------------------------
        
        #print(order,Amp) 
        #fitting  the dataset into the model----------------------------------------------
        #tim1=time.time()
        #print(order)
        #print(Amp.shape)
        
        #tim2=time.time()
        #print('model_training_time',tim2-tim1)
        #saving the updated dataset into the current directory for later use--------------
        #delete these file after you change the the thres value or started a new process
        if(i>0 and i<2):
          t_ser = np.array([[i],[i+1]])
          t_ser = reg.fit_fun(t_ser)
        if(i>1):
          t_ser = np.array([[i-1],[i],[i+1]])
          t_ser = reg.fit_fun(t_ser)
        if(i==0):
          t_ser = np.array([[i+1]])
          t_ser = reg.fit_fun(t_ser)
        np.save("temp_train_data_Order",order)
        np.save("temp_train_data_Amp",Amp)
        dam = Ridge(alpha=alpha1)
        if(i>0 and i<2):
          Amp = Amp[-2:]
          dam.fit(t_ser, Amp)
        if(i>1):
          Amp = Amp[-3:]
          dam.fit(t_ser, Amp)
        if(i==0):
          dam.fit(t_ser, Amp)
        
        model = dam 
        i = i+1
        #returns the two matrices with position of order amplitudes
        #return ls1 , ls2  
    def get_tn(order):
      global model
      global shape1 ,shape2 , size1 , size2
      global i 
      from sklearn.metrics import r2_score
      #possible time steps with trainig greater than 3
      series = np.array([2.1 , 2.2 , 2.3 , 2.4 ,2.5 , 2.6 , 2.7 ,2.8 , 2.9 ,3, 3.1, 3.3,3.4 , 3.5, 3.6, 3.7, 3.8 , 3.9 , 4 ,4.1 , 4.2 , 4.3 , 4.5,4.6,4.7,4.8,4.9 , 5,5.1 ,5.2 , 5.3 ,5.4,5.5, 5.6, 5.7 ,5.8, 5.9, 6 ,6.1, 6.2, 6.3,6.4 ,6.5, 6.7 ,6.8, 7.1, 7.2 , 7.3 ,7.4, 7.5 , 7.6, 7.7,7.8, 7.9, 8 ,8.1, 8.3 , 8.4, 8.5, 8.6, 8.7 ,8.8, 8.9,  9 , 9.1 , 9.2 , 9.3 , 9.4 ,9.5 , 9.6 , 9.7 , 9.8,9.9,10 ,10.1 ,10.2, 10.3 , 10.4, 10.5 , 10.6, 10.7 , 10.9 , 10.8 , 10.9, 11.1 , 11.2 ,11.3 , 11.5 ,12])
      score = np.array([0])
      for i in series:
        output = np.double(model.predict(reg.fit_fun(np.array([[i]]))))
        t1 = output[0,:size1].reshape(shape1)
        t2 = output[0,size1:].reshape(shape2)
        ls2=np.load('ls2.npy')
        ls4=np.load('ls4.npy')
        #print(ls2.size)
        if ls2.size==1:
            op1=[]
        else:
            #print('what are you doing here?')
            op1=t1[ls2[:,0],ls2[:,1]]
        if ls4.size==1:
            op2=[]
        else:
            op2=t2[ls4[:,0],ls4[:,1],ls4[:,2],ls4[:,3]]
        output=np.concatenate((np.array(op1),np.array(op2)))
        output=output.reshape(1,output.size)
        
        ##here we use that prediction thing
        #output = output*position
        #output = output[np.abs(output)>0]
        
        
        y = (np.reshape((order), (order.size)))
        x = np.reshape((output), (output.size)) 
        score = np.array(np.append(score , r2_score(x, y)))
        
        
        

      print(score)
      score = score[1:]
      pos = np.max(score)
      val = score<pos

      val =  (-1*val)+1
      series = series*val
      series = series[series>0]
      series = series[0]
      if ( order.size > 300 and series>6):
      	series = series - 0.6
      if ( order.size > 300 and series<6):
        series = series - 0.1
      if(series>7 and i<5 ):
      	series = series-0.9
      print(series)
      #print("Ideal time iteration is estimated to be "+ str(series) +" you can try values in between "+ str(series-1)+" and "+ str(series+1) +" to check for increased accuracy, this value can be passes as 4th quantitiy in reg.predict ")
      return np.double(model.predict(reg.fit_fun(np.array([[series]]))))
          
    

    def reg_predict(X):
        global model
        global shape1 ,shape2 , size1 , size2
        #X =np.reshape(X.flatten(),(1,X.size))
        #time_series_prediction using r_2
        pred = reg.get_tn(X)
        t1_ten = pred[0,:size1].reshape(shape1)
        t2_ten = pred[0,size1:].reshape(shape2)
        #pred = np.reshape(pred,shape)
        return t1_ten , t2_ten
    
    
    def accu(X,Y):
        #calculating accuracy of original and predicted tensor, run only after reg_predict
        from sklearn.metrics import r2_score
        size = X.size
        x1 = np.reshape(X, (size))
        y1 = np.reshape(Y, (size))
        print(r2_score(x1, y1))
        return (r2_score(x1, y1))

#def prediction_of_all(t1,t2,thres):
     
