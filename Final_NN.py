'''
--------------------------------------------------------------------------
  Module     :     <Simple (Convolutional) Neural Network Classification>
  Author      :    <Le Trong Hieu>                    
  Date        :    <2018-03-01>
  University  :    <Da Nang University of Technology>
--------------------------------------------------------------------------
'''


import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
from numpy.lib.stride_tricks import as_strided
from PIL import Image, ImageOps
from skimage import transform
import os
import cv2
import time
from scipy import ndimage
import sys
import numba as nb
from numba import jit, float32, njit, prange, stencil
from tqdm import tqdm
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Layout, Figure, Scatter
import plotly as py
from scipy.signal import convolve2d
import pickle
 
#   pylint: disable = E1101

listInit = ["lecun_normal", "lecun_uniform", "glorot_normal", "glorot_uniform"]
listOptimizer = ["sgd","adadelta","momentum","nag","adam","adamax", "rmsprop","adagrad"]
listAlpha = [0.01,0.001,0.0001]
 
def printGraph(R):
    N = len(R[0])
    x = np.array(range(N))
    trainA = np.asarray([i[0] for i in R[0]])/ 100
    trainE = [i[1] for i in R[0]]
    validA = np.asarray([i[0] for i in R[1]])/ 100
    validE = [i[1] for i in R[1]]
 
    trace0 = Scatter(
        x = x,
        y = trainA,
        line = dict(
            color = ('rgb(22, 96, 167)'),
            width = 2),
        name = 'Training Accuracy'
    )
    trace1 = Scatter(
        x = x,
        y = trainE,
        line = dict(
            color = ('rgb(22, 96, 167)'),
            width = 2,
            dash = 'dot'),
        name = 'Training Error'
    )
 
    trace2 = Scatter(
        x = x,
        y = validA,
        line = dict(
            color = ('rgb(205, 12, 24)'),
            width = 2),
        name = "Validate Accuracy"
    )
 
    trace3 = Scatter(
        x = x,
        y = validE,
        line = dict(
            color = ('rgb(205, 12, 24)'),
            width = 2,
            dash = 'dot'),
        name = "Validate Error"
    )
 
    data = [trace0, trace1, trace2, trace3]
    layout = Layout(
        title = R[2],
    )
    plot(Figure(data=data, layout = layout), filename= R[2] + ".html")

    print(len(R[3]))
    pickling_on = open(R[2] + ".pickle","wb")
    pickle.dump(R[3], pickling_on)
    pickling_on.close()
   
def emptyList(lenght):
    return [None for _ in range(lenght)]
 
def T(x):
    return np.transpose(x)
 
@jit(float32[:,:](float32[:,:],float32[:,:]), fastmath = True)
def mul(a, b):
    return np.matmul(a, b)
 
def ZL(x):
    return np.zeros_like(x)

@jit(float32[:,:,:](float32[:,:,:]), fastmath = True)
def zoom(x):
    s, d = x.shape[0], x.shape[2]
    y = np.zeros((s*2, s*2, d), dtype=np.float32)
    for c in range(d):
        y[:, :, c] = ndimage.zoom(x[:, :, c], 2, order=0)
    return y
 
@jit(float32[:,:](float32[:,:]))
def rotate(x):
    return np.fliplr(np.flipud(x))
 
@jit(float32[:,:](float32[:,:,:],float32[:,:,:]), fastmath = True)
def conv3D(input, weight):
    S = input.shape[0] - weight.shape[0] + 1
    res = np.zeros((S, S), dtype=np.float32)
    for i in range(input.shape[2]):
        res += convolve2d(input[...,i],weight[...,i],mode='valid')
    return res
 
 
class Data:
    def __init__(self, path, sizeImage):
        Images = os.listdir(path)
        trainImages = []
        trainLabels = []
        l = np.array([[0, 1], [1, 0]], dtype=np.float32).reshape(2, 2, 1)
        for image in tqdm(Images, leave=False):
            pathImage = os.path.join(path, image)
            img = cv2.imread(pathImage)
            trainImages.append(transform.resize(img, (sizeImage, sizeImage, 3)))
            if (image.find("cat")) != -1:
                trainLabels.append(l[1])
            else:
                trainLabels.append(l[0])
        self.data = (np.array(trainImages), np.array(trainLabels))
 
    def standardize(self):
        x = self.data[0]
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        self.data = (x, self.data[1])
 
    @staticmethod
    def shuff(trainData, trainLabels):
        indices = np.array(range(len(trainData))).reshape((len(trainData),))
        random.shuffle(indices)
        trainData = trainData[indices, ...]
        trainLabels = trainLabels[indices, ...]
        return (trainData, trainLabels)
 
    def split(self, fraction=0.8):
        data = self.data
        index = int(data[0].shape[0]*fraction)
        self.data = (data[0][:index], data[1][:index],
                     data[0][index:], data[1][index:])
 

class Function:
    def __init__(self, f):
        self.function = f.lower()
 
    def __str__(self):
        if self.function == None:
            return "None"
        else:
            return "%s" % (self.function)
 
    def calculate(self, input):
        if self.function == "sigmoid":
            input[input>20] = 20
            input[input<-20] = -20
            return 1/(1 + np.exp(-input))
        elif self.function == "relu":
            R = np.copy(input)
            R[R < 0] = 0
            return R
        elif self.function == "vectorize":
            S, D = input.shape[0], input.shape[2]
            R = np.zeros((S*S, D), dtype=np.float32)
            for i in range(S*S):
                for k in range(D):
                    R[i, k] = input[i % S, i//S, k]
            return R.reshape(-1, 1)
        else:
            f = np.mean if self.function == "mean" else np.max
            blockShape = (2, 2)
            inputShape = np.array(input.shape)
            blockShape = np.array(blockShape)
            input = np.ascontiguousarray(input)
            newShape = (inputShape[0] // blockShape[0], inputShape[1] // blockShape[1], inputShape[2]) + tuple(blockShape)
            newStrides = tuple(input.strides[:-1] * blockShape) + tuple([input.strides[-1], input.strides[0], input.strides[1]])
            output = as_strided(input, shape=newShape, strides=newStrides)
            for _ in range(2):
                output = f(output, axis=-1)
            return np.asarray(output, dtype= np.float32)
    @jit
    def deviation(self, N, A, D):
        R = np.copy(N)
        if self.function == "sigmoid":
            return A * (1-A) * D
        elif self.function == "relu":
            R[R > 0] = 1
            R[R <= 0] = 0
            return R * D
        elif self.function == "vectorize":
            s, d = A.shape[0], A.shape[-1]
            R = np.zeros((s, s, d), dtype=np.float32)
            for a in range(s):
                for b in range(s):
                    for c in range(d):
                        R[a, b, c] = D[c*s*s + b*s + a, 0]
            return R
        elif self.function == "mean":
            return D /4
        else:
            # max
            Z = zoom(A)
            R[Z != N] = 0
            R[Z== N] = 1
            return R * D
 

class Parameter:
    alpha = 0.01
    decay = 0.9
    epsilon = 10**(-8)
    beta1 = 0.9
    beta2 = 0.99
 
    def __init__(self, typeParameter, shapeParameter, optimizer, initializer, alpha):
        if initializer not in {'lecun_uniform', 'lecun_normal',
                               'glorot_uniform', 'glorot_normal',
                               'he_uniform', 'he_normal'}:
            raise ValueError("Invalid initializer")
        self.alpha = alpha
        self.optimizer = optimizer
        self.initializer = initializer
        self.value = self.initValue(typeParameter, shapeParameter)
        self.D = ZL(self.value)
        if self.optimizer == "adadelta":
            self.Eg2 = ZL(self.value)
            self.ED2 = ZL(self.value)
        elif self.optimizer == "adam" or self.optimizer == "adamax":
            self.m = ZL(self.value)
            self.v = ZL(self.value)
            self.t = 0
        elif self.optimizer == "momentum":
            pass
        elif self.optimizer == "nag" or self.optimizer == "sgd":
            pass
        elif self.optimizer == "adagrad" or self.optimizer == "rmsprop":
            self.cache = ZL(self.value)
       
        else:
            raise RuntimeError("Not found optimizer")
 
    def initValue(self, typeParameter, shapeParameter):
        if typeParameter == "bias":
            if len(shapeParameter) == 4:
                return np.zeros((shapeParameter[3]), dtype = np.float32)
            else:
                return np.zeros((shapeParameter[0], 1), dtype = np.float32)
        else:
            fans = self.getFans(shapeParameter)
        if self.initializer == 'lecun_uniform':
            return self.generateDitribution('uniform', shapeParameter, fans, 'fanIn', 3.)
        elif self.initializer == 'lecun_normal':
            return self.generateDitribution('normal', shapeParameter, fans, 'fanIn', 1.)
        elif self.initializer == 'glorot_uniform':
            return self.generateDitribution('uniform', shapeParameter, fans, 'fanAvg', 3.)
        elif self.initializer == 'glorot_normal':
            return self.generateDitribution('normal', shapeParameter, fans, 'fanAvg', 1.)    
        elif self.initializer == 'he_uniform':
            return self.generateDitribution('uniform', shapeParameter, fans, 'fanIn', 6.)
        else:
            # he_normal
            return self.generateDitribution('normal', shapeParameter, fans, 'fanIn', 2.)
           
    def generateDitribution(self, distribution, shape, fans, mode, scale):
        n = (fans[0]+fans[1])/2 if mode == "fanAvg" else fans[mode != "fanIn"]
        s = np.sqrt(scale / n)
        if distribution == "normal":
            return np.random.normal(loc = 0, scale = s, size = shape)
        else:
            return np.random.uniform(low = -s, high = s, size = shape)
       
    def getFans(self, shape):
        if len(shape) == 2:
            return (shape[1], shape[0])
        else:
            return (np.prod(shape[:3]), shape[-1])
    # @jit
    def learn(self, g):
        if self.optimizer == "adadelta":
            self.Eg2 = self.decay * (self.Eg2) + (1-self.decay) * g**2
            self.D = - np.sqrt(self.ED2 + self.epsilon) * g / np.sqrt(self.Eg2 + self.epsilon)
            self.ED2 = self.decay * self.ED2 + (1-self.decay) * self.D * self.D
        elif self.optimizer == "momentum":
            self.D = self.decay * self.D - self.alpha * g
           
        elif self.optimizer == "adam":
            self.t += 1
            self.m = self.beta1 * self.m + (1-self.beta1) * g
            self.v = self.beta2 * self.v + (1-self.beta2) * g**2
            mHat = np.divide(self.m , (1 - np.power(self.beta1, self.t)))
            vHat = np.divide(self.v , (1 - np.power(self.beta2, self.t)))
            self.D = - np.divide((self.alpha * mHat) , (np.sqrt(vHat) + self.epsilon))
        elif self.optimizer == "adamax":
            self.t += 1
            self.m = self.beta1 * self.m + (1-self.beta1) * g
            self.v = np.maximum(self.beta2 * self.v, np.abs(g))
            self.D = - self.alpha * self.m / ((1 - np.power(self.beta1, self.t)) * (self.v+self.epsilon))
        elif self.optimizer == "nag":
            pD = self.D
            self.D = self.decay * self.D - self.alpha * g
            self.D = - self.decay * pD + (1+ self.decay)* self.D
        elif self.optimizer == "adagrad":
            self.cache += np.multiply(g,g)
            self.D = - np.divide(np.multiply(self.alpha, g),
                                 np.sum((np.sqrt(self.cache),
                                         self.epsilon)
                                       )
                                )
        elif self.optimizer == "rmsprop":
            self.cache = self.decay * self.cache + (1 - self.decay) * g**2
            self.D = - self.alpha * g / (np.sqrt(self.cache) + self.epsilon)
        elif self.optimizer == "sgd":
            self.D = - self.alpha * g
        self.value += self.D
 

class Layer:
    def __init__(self, typeLayer, initializer = None, optimizer = None, shapeParameter = None, activation = None, alpha = None):
        self.typeLayer = typeLayer
        self.activation = Function(activation)
        self.learnable = (typeLayer == "F" or typeLayer == "C")
        if self.learnable:
            self.weight = Parameter("weight", shapeParameter, optimizer, initializer, alpha)
            self.bias = Parameter("bias", shapeParameter, optimizer, initializer,alpha)
 
    def __str__(self):
        s = "type: %s, activation: %s" % (self.typeLayer, self.activation.__str__())
        if self.learnable:
            s += ", weight shape: %s, bias shape: %s" %(self.weight.value.shape, self.bias.value.shape)
        return s
 
    def learn(self, DW, DB):
        self.weight.learn(DW)
        self.bias.learn(DB)
 
    # @jit(nopython=True)
    def calculate(self, inp):
        if self.typeLayer == "C":
            S, D = inp.shape[0] - self.weight.value.shape[0] + 1, self.weight.value.shape[3]
            N = np.zeros((S, S, D), dtype = np.float32)
            for q in range(D):
                N[..., q] = conv3D(inp, self.weight.value[..., q]) + self.bias.value[q]
            return (N, self.activation.calculate(N))
        elif self.typeLayer == "F":
            N = (mul(self.weight.value, inp) + self.bias.value)
            return (N, self.activation.calculate(N))
        else:
            return (inp, self.activation.calculate(inp))
 
 
class Network:
    def __init__(self, optimizer, initializer, alpha):
        self.layers = []
        self.L = 0
        self.optimizer = optimizer.lower()
        self.initializer = initializer.lower()
        self.forwardT = 0.
        self.backwardT = 0.
        self.forwardN = 0
        self.backwardN = 0
        self.alpha = alpha
 
    def __str__(self):
        s = "Initializer: %s, optimizer: %s, number Parameter: %d \n" %(self.initializer, self.optimizer, self.numParameter())
        for i in range(self.L):
            s += "Layer: %d, %s \n" %(i, self.layers[i].__str__())
        return s
 
    def add(self, typeLayer, shape = None, activation = "sigmoid", shapeInput = None):
        if typeLayer == "C":
            shapeWeight = (shape[0], shape[1],self.layers[-2].weight.value.shape[3], shape[2]) if shapeInput == None else (shape[0], shape[1], shapeInput[2], shape[2])
            self.layers.append(Layer(
                                    typeLayer,
                                    optimizer = self.optimizer,
                                    shapeParameter = shapeWeight,
                                    activation = activation,
                                    initializer = self.initializer,
                                    alpha = self.alpha))
        elif typeLayer == "P":
            self.layers.append(Layer(
                                    typeLayer,
                                    activation = activation))
        elif typeLayer == "V":
            self.layers.append(Layer(typeLayer,
                                    activation = "vectorize"))
        else:
            self.layers.append(Layer(
                                    typeLayer,
                                    optimizer = self.optimizer,
                                    shapeParameter = shape,
                                    activation = activation,
                                    initializer = self.initializer,
                                    alpha=self.alpha))
        self.L += 1
 
    def trans(self, i):
        return (self.layers[i+1].typeLayer, self.layers[i].typeLayer)
 
    def accuracy(self, testData, testLabels):
        correctSample = 0
        meanError = 0.
        for indexSample in tqdm(range(len(testData)),leave=False):
            (_, A, _) = self.forward(testData[indexSample, ...])
            e = A[-1] - testLabels[indexSample, ...]
            meanError += mul(T(e), e)[0, 0]
            correctSample += (np.argmax(A[-1]) ==
            np.argmax(testLabels[indexSample, ...]))
 
        return (correctSample/len(testData) * 100, meanError / len(testData))
   
    def timeAvg(self):
        print("Avg forward time: %4.2f" %(self.forwardT/ self.forwardN))
        print("Avg backward time: %4.2f" %(self.backwardT/ self.backwardN))
        self.forwardN = 0
        self.forwardT = 0.
        self.backwardN = 0
        self.backwardT = 0.
 
    def numParameter(self):
        res = 0
        for i in range(self.L):
            if self.layers[i].learnable:
                if self.layers[i].typeLayer == 'C':
                    res += np.prod(self.layers[i].weight.value.shape[:3]) + self.layers[i].bias.value.shape[0]
                else:
                    res += np.prod(self.layers[i].weight.value.shape) + self.layers[i].bias.value.shape[0]
        return res
 
    def forward(self, data):
        t =time.time()
        A = emptyList(self.L)
        N = emptyList(self.L)
        T1 = np.zeros((self.L))
        for i in range(self.L):
            t2 = time.time()
            if i == 0:
                (N[0], A[0]) = self.layers[0].calculate(data)
            else:
                (N[i], A[i]) = self.layers[i].calculate(A[i-1])
            T1[i] = time.time() - t2
        self.forwardN += 1
        self.forwardT += time.time() - t
        return (N, A, T1)
   
    def backward(self, N, A, input, target):
        t = time.time()
        DA = emptyList(self.L)
        DAf = emptyList(self.L)
        DW = emptyList(self.L)
        DB = emptyList(self.L)
        T2 = np.zeros(self.L)
        for i in range(self.L):
            if self.layers[i].learnable:
                DW[i] = ZL(self.layers[i].weight.value)
                DB[i] = ZL(self.layers[i].bias.value)
 
        e = target - A[self.L-1]
        t2 = time.time()
        DA[-1] = -2 * self.layers[-1].activation.deviation(N[-1], A[-1], e)
        DW[-1] = mul(DA[-1], T(A[-2]))
        DB[-1] = DA[-1]
        T2[self.L-1] = time.time() - t2
        for i in range(self.L-2, -1, -1):
            t2 = time.time()
            if self.trans(i) == ('F', 'F'):
                DA[i] = self.layers[i].activation.deviation(N[i], A[i], mul(T(self.layers[i+1].weight.value), DA[i+1]))
                DW[i] = mul(DA[i], T(A[i-1]))
                DB[i] = DA[i]
 
            elif self.trans(i) == ('F', 'V'):
                DA[i] = mul(T(self.layers[i+1].weight.value), DA[i+1])
 
            elif self.trans(i) == ('V', 'P'):
                DA[i] = self.layers[i+1].activation.deviation(N[i], A[i], DA[i+1])
               
            elif self.trans(i) == ('P', 'C'):
                inputLayer = input if i == 0 else A[i-1]
                inputLayer = np.asarray(inputLayer, dtype=np.float32)
 
                DA[i] = self.layers[i+1].activation.deviation(N[i+1], A[i+1], zoom(DA[i+1]))
                DAf[i] = self.layers[i].activation.deviation(N[i], A[i], DA[i])
 
                for p in range(self.layers[i].weight.value.shape[2]):
                    rot = rotate(inputLayer[..., p])
                    for q in range(self.layers[i].weight.value.shape[3]):
                        DW[i][...,p,q] = convolve2d(rot, DAf[i][...,q], mode='valid')
 
                for p in range(self.layers[i].weight.value.shape[3]):
                    DB[i][p] += DAf[i][..., p].sum()
            else:
                # from C to P
                DA[i] = ZL(A[i])
                for p in range(A[i].shape[2]):
                    for q in range(A[i+1].shape[2]):
                        DA[i][..., p] += convolve2d(DAf[i+1][...,q],
                        rotate(self.layers[i+1].weight.value[...,p,q]), mode='full')
                       
            T2[i] = time.time() - t2
        self.backwardN += 1
        self.backwardT += time.time() - t
        return (e, DW, DB,T2)
 
    def accumulateWeight(self, DW, DB, dw, db):
        for i in range(self.L):
            if self.layers[i].learnable:
                DW[i] += dw[i]
                DB[i] += db[i]
        return (DW, DB)
    @jit(fastmath = True)
    def update(self, DW, DB, batchSize):
        for i in range(self.L):
            if self.layers[i].learnable:
                self.layers[i].learn(DW[i]/batchSize, DB[i]/batchSize)
    @jit
    def initDeviationWeight(self):
        DW = emptyList(self.L)
        DB = emptyList(self.L)
        for i in range(self.L):
            if self.layers[i].learnable:
                DW[i] = ZL(self.layers[i].weight.value)
                DB[i] = ZL(self.layers[i].bias.value)
        return (DW, DB)
    @jit(fastmath = True)
    def calculate(self, maxLoop, batchSize, data):
        print(self.__str__())
        trainData, trainLabels, testData, testLabels = data.data
        numBatch = len(trainData) // batchSize
        trainLog = []
        validLog = []
        layerLog = []
        trainAcc = self.accuracy(trainData, trainLabels)
        validateAcc = self.accuracy(testData, testLabels)
        print("TRAINING: Accuracy %4.3f %% Error: %4.3f" % trainAcc)
        print("VALIDATE: Accuracy %4.3f %% Error: %4.3f" % validateAcc)        
        print("-------------")
        trainLog.append(trainAcc)
        validLog.append(validateAcc)
        for iterator in range(maxLoop):
            print("************")
            print("Loop", iterator+1, "/", maxLoop)
            (trainData, trainLabels) = Data.shuff(trainData, trainLabels)
            for indexBatch in range(numBatch):
                print("Batch", indexBatch+1, "/", numBatch)
                (DW, DB) = self.initDeviationWeight()
                for indexInBatch in tqdm(range(batchSize),leave=False):
                    indexSample = indexInBatch + batchSize * indexBatch
                    (N, A, _) = self.forward(trainData[indexSample, ...])
                    (_, dw, db, _) = self.backward(N, A, trainData[indexSample, ...],
                                                trainLabels[indexSample, ...])
                    (DW, DB) = self.accumulateWeight(DW, DB, dw, db)
                self.update(DW, DB, batchSize)
                trainAcc = self.accuracy(trainData, trainLabels)
                validateAcc = self.accuracy(testData, testLabels)
                print("TRAINING: Accuracy %4.3f %% Error: %4.3f" % trainAcc)
                print("VALIDATE: Accuracy %4.3f %% Error: %4.3f" % validateAcc)
                self.timeAvg()
                print("-------------")
                trainLog.append(trainAcc)
                validLog.append(validateAcc)
                layerLog.append(self.layers)
        s = "%s, %s," %(self.initializer, self.optimizer) + str(self.layers[-1].weight.alpha) + str(self.numParameter())
        s += " %d samples, %s loops, %d batchSize" %(len(trainData)/0.8, maxLoop, batchSize)
        for i in range(self.L):
            s+= self.layers[i].typeLayer
            if self.layers[i].learnable:
                s += str(self.layers[i].weight.value.shape[0])
        printGraph((trainLog, validLog, s, layerLog))
       
 
def CNN(sizeImage, cnvLayers, spatialSize = 5, depthOutput = 8,
        FCLayers = 2, optimizer = 'adadelta', initializer = 'lecun_normal', alpha = 0.01, neuron = None):
    net = Network(optimizer = optimizer, initializer = initializer, alpha = alpha)
    currentSize = [sizeImage, 3]
 
    net.add('C', shape = (spatialSize, spatialSize, depthOutput), activation = 'relu', shapeInput=(sizeImage, sizeImage,3))
    currentSize = [currentSize[0] - spatialSize + 1, depthOutput]
    net.add('P', activation = 'Max')
 
    if currentSize[0] % 2 != 0:
        raise ValueError('unvalid size')
    currentSize[0] //= 2
 
    for _ in range(cnvLayers-1):
        depthOutput *= 2
        net.add('C', shape = (spatialSize, spatialSize, depthOutput), activation='relu')
        currentSize = [currentSize[0] - spatialSize + 1, depthOutput]
        net.add('P', activation = 'Max')
        if currentSize[0] % 2 !=0:
            raise ValueError('unvalid size')
        currentSize[0] //= 2
        print(currentSize)
 
    net.add('V')
    vectorSize = int(currentSize[0]**2 * currentSize[1])
    if neuron == None:
        neuron = int(2**int(np.log2(int(currentSize[0]**2 * currentSize[1]))))
   
    net.add('F', (neuron, vectorSize), 'relu')
   
    for _ in range(FCLayers-2):
        net.add('F', (neuron//2, neuron), 'relu')
        neuron //= 2
 
    net.add('F', (2, neuron), 'sigmoid')
    return net
 
def NN(sizeImage,FCLayers = 2, optimizer = 'adadelta',
       initializer = 'lecun_normal', alpha = 0.01, neuron = None):
    net = Network(optimizer = optimizer, initializer = initializer, alpha = alpha)
    net.add('V')
    if neuron == None:
        neuron = int(2**int(np.log2(sizeImage * sizeImage * 3)))
    input = int(sizeImage * sizeImage * 3)
    net.add('F', (neuron, input), 'relu')
    for _ in range(FCLayers-2):
        net.add('F', (neuron//2, neuron), 'relu')
        neuron //= 2
 
    net.add('F', (2, neuron), 'sigmoid')
    return net
 
data = Data(path = "C:\\train200", sizeImage = 52)
data.standardize()
data.data = Data.shuff(data.data[0], data.data[1])
data.split(0.8)
 
# nn = Network(alpha = 0.001, optimizer = "rmsprop", initializer = "lecun_normal")
# nn.add("V")
# nn.add("F", (512, 7500), activation="relu")
# nn.add("F", (512, 512), activation="relu")
# nn.add("F", (256, 512), activation="relu")
# nn.add("F", (256, 256), activation="relu")
# nn.add("F", (256, 256), activation="relu")
# nn.add("F", (2, 256), activation="sigmoid")
# R = nn.calculate(maxLoop = 5, batchSize = 40, data = data)

net = CNN(sizeImage = 52, 
        initializer = 'glorot_uniform',
        optimizer='adadelta', alpha= 0.01,
        cnvLayers = 3, spatialSize= 5, depthOutput= 4,
        FCLayers = 2, neuron = 2048)
net.calculate(maxLoop = 10, batchSize = 80, data = data)

# net = NN(sizeImage = 52, 
#         initializer= 'glorot_uniform',
#         optimizer='rmsprop', alpha= 0.0001,
#         FCLayers = 4, neuron = 1024)
# net.calculate(maxLoop = 3, batchSize = 80, data = data)

# for opti in listOptimizer:
#     for alpha in listAlpha:
#         net = CNN(sizeImage = 52, cnvLayers = 2, spatialSize= 5, 
#             depthOutput= 8, optimizer=opti,alpha= alpha, initializer = 'glorot_uniform')
#         net.calculate(maxLoop = 10, batchSize = 32, data = data)