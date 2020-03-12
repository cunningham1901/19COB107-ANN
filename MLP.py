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

    #Set weights at given edgelayer
    def setWeights(self, edgeLayerIndex, weightMatrix):
        self.edgeLayers[edgeLayerIndex].weights = weightMatrix

    #Set biases at given nodeLayer
    def setBiases(self, layerIndex, biasVector):
        #Check correct number of inputs given
        if len(biasVector) != len(self.nodeLayers[layerIndex].nodeBiases):
            return False
        else:
            self.nodeLayers[layerIndex].nodeBiases = biasVector


    #Run a forward pass through the network
    def forwardPass(self):

        #Calculate the activations for the next node later
        def calculateForwardValues(values, weights, biases):
            def sigmoid(x):
                return 1/(1+exp(-x))
            vectorized_sigmoid = np.vectorize(sigmoid)
            #activation vector = (matrix-vector product weights and values)+biases
            return list(vectorized_sigmoid(np.dot(weights, values) + biases))

        #Calculate activation vector for each layer in network
        for i in range(len(self.edgeLayers)):
            currValues = np.array(self.nodeLayers[i].nodeValues).transpose()
            currWeights = np.array(self.edgeLayers[i].weights)
            currBiases = np.array(self.nodeLayers[i+1].nodeBiases).transpose()
            nextValues = calculateForwardValues(currValues, currWeights, currBiases)
            self.nodeLayers[i+1].nodeValues = nextValues

    #Run a backward pass through the network
    def backwardPass(self, correct):
        lastLayer = len(self.nodeLayers)-1
        #Calculate for each node in output layer
        for i in range(len(self.nodeLayers[lastLayer].nodeValues)):
            activation = self.nodeLayers[lastLayer].nodeValues[i]
            self.nodeLayers[lastLayer].nodeDeltas[i] = activation*(correct-activation)*(1-activation)

        #Calculate iDelta for hidden layers
        for layer in range(lastLayer, 1, -1):
            iNodes = len(self.nodeLayers[layer-1].nodeValues)
            jNodes = len(self.nodeLayers[layer].nodeValues)
            weights = self.edgeLayers[layer-1].weights
            #For each node in layer i
            for i in range(iNodes):
                activation = self.nodeLayers[layer-1].nodeValues[i]
                sumWeightsDeltas = 0
                #Get sum of appopriate weights and deltas in layer j
                for j in range(jNodes):
                    delta = self.nodeLayers[layer].nodeDeltas[j]
                    sumWeightsDeltas += weights[j][i]*delta
                self.nodeLayers[layer-1].nodeDeltas[i] = sumWeightsDeltas*activation*(1-activation)

    def updateWeights(self, rho):
        #Calculate new weights, start a
        for x in range(len(self.edgeLayers)):
            #Generate new weight matrix for edgelayer x (between nodelayer x and x+1)
            currWeightMatrix = self.edgeLayers[x].weights
            newWeightMatrix = []
            #weights between i,j:
            for j in range(len(self.nodeLayers[x+1].nodeValues)):
                currWeights = currWeightMatrix[j]
                newWeights = []
                deltaJ = self.nodeLayers[x+1].nodeDeltas[j]
                #for each node j, update input weights from nodes i
                for i in range(len(self.nodeLayers[x].nodeValues)):
                    activationI = self.nodeLayers[x].nodeValues[i]
                    currWeight = currWeights[i]
                    newWeights.append(currWeight + (rho * deltaJ * activationI))
                newWeightMatrix.append(newWeights)
                #update the bias for node j
                currBias = self.nodeLayers[x+1].nodeBiases[j]
                self.nodeLayers[x+1].nodeBiases[j] = currBias + (rho * deltaJ)

            self.edgeLayers[x].weights = newWeightMatrix

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
        self.nodeDeltas=[]
        #Setup with blank data
        for i in range(numNodes):
            self.nodeValues.append(0)
            self.nodeBiases.append(0)
            self.nodeDeltas.append(0)

    #Set the properties of a node
    def setNode(self, nodeIndex, value, bias):
        self.nodeValues[nodeIndex] = value
        self.nodeBiases[nodeIndex] = bias

    #Print a textual representation of the node
    def printSelf(self):
        print("Layer %d" %(self.index))
        for i in range(len(self.nodeValues)):
            print("  Node %d with value %f (bias: %f, delta: %f)"%(i, self.nodeValues[i], self.nodeBiases[i], self.nodeDeltas[i]))

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

    #Set edges values
    def setEdge(self, indexI, indexJ, weight):
        self.weights[indexJ][indexI] = weight

    #Print a textual representation of the edges
    def printSelf(self):
        print("Edges between layers %d and %d"%(self.inputIndex, self.outputIndex))
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                print("  %d -> %d (weight: %f)"%(j, i, self.weights[i][j]))



#an example MLP, 3 layers, 2 input, 1 output
# b = MLP(4, [2,3,2,1])
# b.setRandomWeightsBiases()
#
#
# b.setInputs([1,0])
#
# b.printSelf()


# b.setBiases(1, [2,-4,5])
# b.setBiases(2,[1,-6])
# b.setBiases(3, [-3.92])
#
# weight0 = [[1,4],
#            [2,3],
#            [1,1]]
# b.setWeights(0, weight0)
# weight1 = [[3,4,5],
#            [6,5,2]]
# b.setWeights(1,weight1);
# weight2 = [[2,4]]
# b.setWeights(2,weight2);
# b.forwardPass()
# b.backwardPass(1)
# b.updateWeights(0.1)
# b.printSelf()
