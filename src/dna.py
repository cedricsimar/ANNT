
from settings import Settings
from vertex import Vertex
from edge import Edge

class DNA:

    """
    DNA class contains the graph-like structure representation of the neural network
    ans implements all the necessary operations that can be performed on a DNA
    """

    def __init__(self, input_shape, output_shape, multiclass=False):
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.multiclass = multiclass

        self.vertex_id = 0
        self.edge_id = 0

        self.vertices = {}
        self.edges = {}

        self.num_fully_connected_layers = 0 # limit to 2 FC to limit computation time

        self.create_primitive_structure()
    

    def __str__(self):

        net_str = ""

        for edge in self.edges.keys():

            if self.edges[edge].type == Settings.FULLY_CONNECTED:
                net_str += "Fully connected layer "
            else:
                net_str += "Convolutional layer "
            
            net_str += "from vertex " + str(self.edges[edge].from_vertex.id)
            net_str += " to vertex " + str(self.edges[edge].to_vertex.id)
            net_str += "\n"
        
        return (net_str)
        

    def create_primitive_structure(self):

        # input vertex = 0, output vertex = 1, edge to output = 0 -> not mutable
    
        # create input vertex
        self.vertices[self.vertex_id] = Vertex(self.vertex_id, mutable=[False, True, False])
        self.vertex_id += 1

        # create output vertex
        self.vertices[self.vertex_id] = Vertex(self.vertex_id, mutable=[False, False, False])
        self.vertex_id += 1

        # create fc edge between input and output with the number of units equal to the output shape
        self.edges[self.edge_id] = Edge(self.vertices[0], self.vertices[1], units=self.output_shape)
        self.edge_id += 1

        # force the link between input and output (switch on mutability temporarily)
        # switch on mutability
        self.vertices[1].mutable_in = True

        # add the link between the input and output vertices
        self.vertices[0].add_edge_out(self.edges[0])
        self.vertices[1].add_edge_in(self.edges[0])

        # switch back off mutability
        self.vertices[1].mutable_in = False