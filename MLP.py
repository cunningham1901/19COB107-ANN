import numpy as np
from math import exp, sqrt
#Defining the MLP
class MLP:

    def __init__(self, layers, nodesPerLayer):
        self.nodeLayers = []
        self.edgeLayers = []
        #Generate the layers of nodes
        for i in range(layers):
            self.nodeLayers.append(NodeLayer(i, nodesPerLayer[i]))

        #Generate the layers of edges, add them to the list
        for i in range(layers-1):
            self.edgeLayers.append(EdgeLayer(self.nodeLayers[i], self.nodeLayers[i+1]))

        #assign random weights and biases
        self.setRandomWeightsBiases()

    def setInputs(self, inputVector):
        #Check correct number of inputs given
        if len(inputVector) != len(self.nodeLayers[0].nodeValues):
            return False
        else:
            self.nodeLayers[0].nodeValues = inputVector

    def setWeights(self, edgeLayerIndex, weightMatrix):
        self.edgeLayers[edgeLayerIndex].weights = weightMatrix

    def setBiases(self, layerIndex, biasVector):
        #Check correct number of inputs given
        if len(biasVector) != len(self.nodeLayers[layerIndex].nodeBiases):
            return False
        else:
            self.nodeLayers[layerIndex].nodeBiases = biasVector


    def forwardPass(self):

        def calculateForwardValues(values, weights, biases):
            def sigmoid(x):
                return 1/(1+exp(-x))
            vectorized_sigmoid = np.vectorize(sigmoid)
            #matrix-vector product weights and values
            return list(vectorized_sigmoid(np.dot(weights, values) + biases))

        for i in range(len(self.edgeLayers)):
            currValues = np.array(self.nodeLayers[i].nodeValues).transpose()
            currWeights = np.array(self.edgeLayers[i].weights)
            currBiases = np.array(self.nodeLayers[i+1].nodeBiases).transpose()
            nextValues = calculateForwardValues(currValues, currWeights, currBiases)
            self.nodeLayers[i+1].nodeValues = nextValues

    def setRandomWeightsBiases(self):
        #Generate a random weight/bias using numpy gaussian
        for x in range(len(self.edgeLayers)):
            #Generate a new weight matrix or edgelayer x
            newWeightMatrix = []
            #weights into j
            for j in range(len(self.nodeLayers[x+1].nodeValues)):
                noWeightsIn = len(self.nodeLayers[x].nodeValues)
                newWeights = []
                #Generate a weight for each input weight
                for i in range(noWeightsIn):
                    newWeights.append(np.random.normal(loc=0, scale=(1/sqrt(noWeightsIn))))
                newWeightMatrix.append(newWeights)
                #bias of j
                self.nodeLayers[x+1].nodeBiases[j] = np.random.normal(loc=0, scale=(1/sqrt(noWeightsIn)))

            self.edgeLayers[x].weights = newWeightMatrix


    #Print a text representation of the MLP (for debug etc)
    def printSelf(self):
        for i in range(len(self.edgeLayers)):
            self.nodeLayers[i].printSelf()
            self.edgeLayers[i].printSelf()
        self.nodeLayers[len(self.nodeLayers)-1].printSelf()


#A single layer of nodes
#Has an index related to its position in the MLP
class NodeLayer:
    def __init__(self, index, numNodes):
        self.index = index
        self.nodeValues=[]
        self.nodeBiases=[]
        for i in range(numNodes):
            self.nodeValues.append(0)
            self.nodeBiases.append(0)

    def setNode(self, nodeIndex, value, bias):
        self.nodeValues[nodeIndex] = value
        self.nodeBiases[nodeIndex] = bias

    def printSelf(self):
        print("Layer %d" %(self.index))
        for i in range(len(self.nodeValues)):
            print("  Node %d with value %f (bias: %f)"%(i, self.nodeValues[i], self.nodeBiases[i]))

#A layer of edges to sit between layers of nodes
class EdgeLayer:
    def __init__(self, inputLayer, outputLayer):
        self.inputIndex = inputLayer.index
        self.outputIndex = outputLayer.index

        #initialise the weight matrix to 0
        self.weights = []
        for i in range(len(outputLayer.nodeValues)):
            matrixRow = []
            for j in range(len(inputLayer.nodeValues)):
                matrixRow.append(0)
            self.weights.append(matrixRow)

    def setEdge(self, outputNode, inputNode, weight):
        self.weights[indexI][indexJ] = weight


    def printSelf(self):
        print("Edges between layers %d and %d"%(self.inputIndex, self.outputIndex))
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                print("  %d -> %d (weight: %f)"%(j, i, self.weights[i][j]))



#an example MLP, 3 layers, 2 input, 1 output
b = MLP(3, [2,2,1])
b.setInputs([1,0])
b.setBiases(1, [1,-6])
b.setBiases(2, [-3.92])
weight0 = [[3,4],
           [6,5]]
b.setWeights(0,weight0);
weight1 = [[2,4]]
b.setWeights(1,weight1);

b.forwardPass()

b.printSelf()
