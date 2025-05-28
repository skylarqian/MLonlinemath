import numpy as np
l1weights = np.array([[1, 2], [4, 5], [6, 7], [8, 2], [0, 0], [0, 2], [1, 0], [3, 2]]) #8 by 2
l1bias = np.array([1, 1, 2, 3, 4, 3, 2, 1]).reshape(8, 1) #8 by 1
l2weights = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [3, 2, 1, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]]) #3 by 8
l2bias = np.array([-700, -467, 1]).reshape(3, 1) #3 by 1

def backprop(I, Q, real):
    global l1weights, l1bias, l2weights, l2bias
    #real is 0, 1, or 2

    #forward pass
    input = np.array([I, Q])
    inputvector = input.reshape(2, 1) #reshape input to 2 by 1
    print('input vector: ', inputvector)

    layer1out = np.matmul(l1weights, inputvector) + l1bias #matrix mult to get layer1 output of 8 nodes
    print('layer1 output: ', layer1out)

    #relu
    relufunc = np.vectorize(relu)
    reluout = relufunc(layer1out)
    print('relu: ', reluout)

    layer2out = np.matmul(l2weights, reluout) + l2bias #matrix mult to get
    print('layer2 output: ', layer2out)
    layer2outflat = layer2out.reshape(3)

    #softmax
    max = np.max(layer2outflat)
    print('max: ', max)
    denom = np.exp(layer2outflat[0]-max) + np.exp(layer2outflat[1]-max) + np.exp(layer2outflat[2]-max)
    final = np.zeros(3)
    for i in range(3):
        final[i] = np.exp(layer2outflat[i]-max)/denom
    print('final output: ', final)
    predicted = np.argmax(final)
    print('prediction state ', predicted)



    #BACKPROPOGATE
    if (predicted == real):
        return #correct prediction so nothing is changed

    #loss function (softmax -> pre-softmax aka second layer) 
    lossdiff = np.zeros(3) #loss function differential to z dJ/dz
    for i in range(3):
        if (i == predicted): #cannot be real
            lossdiff[i] = sum([final[x] for x in range(3) if x != i])
        elif (i == real):
            lossdiff[i] = 1 - final[i]
        else:
            lossdiff[i] = final[i]
    lossdiff = lossdiff.reshape(3, 1)
    print('lossdiff: ', lossdiff)

    #pre-softmax(second layer) -> first layer
    #dJ/dW2
    layer2andbias = np.append(layer1out.reshape(8), 1).reshape(1, 9)
    print(lossdiff.shape, ' ', layer2andbias.shape)
    weight2diff = np.matmul(lossdiff, layer2andbias) #layer2 weights and bias gradient
    #dJ/dA2
    lossdiffA2 = np.matmul(l2weights.T, lossdiff)

    #first layer -> inputs
    #dJ/W1
    layer1andbias = np.array([I, Q, 1]).reshape(1, 3)
    weight1diff = np.matmul(lossdiffA2, layer1andbias) #layer1 weights and bias gradient

    #adjust weights
    l1weights = weightadjust(64, l1weights, weight1diff[:, :-1]) #shift by 6
    l1bias = weightadjust(64, l1bias, weight1diff[:,-1].reshape(8, 1))
    l2weights = weightadjust(64, l2weights, weight2diff[:,:-1])
    l2bias = weightadjust(64, l2bias, weight2diff[:,-1].reshape(3, 1))
    print('new l1weights: ', l1weights,  ' shape: ', l1weights.shape)
    print('new l1bias: ', l1bias, ' shape: ', l1bias.shape)
    print('new l2weights: ', l2weights, ' shape: ', l2weights.shape)
    print('new l2bias: ', l2bias, ' shape: ', l2bias.shape)




def relu(x):
    if (x >= 0):
        return x
    else:
        return 0
    
def weightadjust(alphainverse, og, gradient):
    return og - gradient/alphainverses
    
backprop(7, 8, 1)