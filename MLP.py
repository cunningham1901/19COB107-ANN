#Defining the MLP

class MLP:
    inputs = []
    output = None

    def __init__(self, layers, nodesPerLayer):
        self.nodeLayers = []
        self.edgeLayers = []
        #Generate the layers of nodes
        for i in range(layers):
            self.nodeLayers.append(NodeLayer(i, nodesPerLayer[i]))

        #Generate the layers of edges
        for i in range(layers-1):
            self.edgeLayers.append(EdgeLayer(self.nodeLayers[i], self.nodeLayers[i+1]))

    def printSelf(self):
        for i in range(len(self.edgeLayers)):
            self.nodeLayers[i].printSelf()
            self.edgeLayers[i].printSelf()
        self.nodeLayers[len(self.nodeLayers)-1].printSelf()


#A single layer of nodes
class NodeLayer:
    def __init__(self, index, numNodes):
        self.index = index
        self.nodes=[]
        for i in range(numNodes):
            self.nodes.append(Node(i))

    def printSelf(self):
        print("Layer %d" %(self.index))
        for i in self.nodes:
            print("  ", end="")
            i.printSelf()

#A layer of edges to sit between layers of nodes
class EdgeLayer:
    def __init__(self, inputLayer, outputLayer):
        self.inputIndex = inputLayer.index
        self.outputIndex = outputLayer.index
        self.groups = []
        for i in inputLayer.nodes:
            self.groups.append(EdgeGroup(i, outputLayer))

    def printSelf(self):
        print("Edges between layers %d and %d"%(self.inputIndex, self.outputIndex))
        for i in self.groups:
            print("  ", end="")
            i.printSelf()

#A group of edges between one node and the following layer
class EdgeGroup:
    def __init__(self, inputNode, outputLayer):
        self.index = inputNode.index
        self.edges=[]
        for i in outputLayer.nodes:
            self.edges.append(Edge(inputNode, i))

    def printSelf(self):
        print("Edges leaving node %d" %(self.index))
        for i in self.edges:
            print("    ", end="")
            i.printSelf()

class Node:

    def __init__(self, index):
        self.index = index

    def printSelf(self):
        print("Node %d"%(self.index))


class Edge:

    def __init__(self, inputNode, outputNode):
        self.inputNode = inputNode
        self.outputNode = outputNode

    def printSelf(self):
        print("%d -> %d" %(self.inputNode.index, self.outputNode.index))

b = MLP(3, [2,3,1])
b.printSelf()
