
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

        # for v in self.vertices.keys():


        for e in self.edges.keys():

            if self.edges[e].type == Settings.FULLY_CONNECTED:
                net_str += "Fully connected layer "
            elif self.edges[e].type == Settings.CONVOLUTIONAL::
                net_str += "Convolutional layer "
            elif self.edges[e].type == Settings.IDENTITY:
                net_str += "Identity layer "
            
            net_str += "from vertex " + str(self.edges[e].from_vertex.id)
            net_str += " to vertex " + str(self.edges[e].to_vertex.id)
            net_str += "\n"
        
        return (net_str)
        

    def create_primitive_structure(self):

        """
        Create a primitive neural network structure that flattens the input and
        uses one dense layer to map the flatten input to the output shape
        """

        # input vertex = 0, output vertex = 1, flatten vertex = 2 
        # identity edge = 0, dense edge = 1
    
        # create input vertex
        self.vertices[self.vertex_id] = Vertex(self.vertex_id, mutable=[False, True, False])
        self.vertex_id += 1

        # create output vertex
        self.vertices[self.vertex_id] = Vertex(self.vertex_id, mutable=[False, False, False])
        self.vertex_id += 1

        # create flatten vertex
        self.vertices[self.vertex_id] = Vertex(self.vertex_id, mutable=[True, True, True], flatten=Settings.FLATTEN)
        self.vertex_id += 1

        # create an identity edge to connect the input with the flatten vertex
        self.edges[self.edge_id] = Edge(self.edge_id, self.vertices[0], self.vertices[2], type=Settings.IDENTITY)
        self.edge_id += 1

        # create fc edge between the flatten vertex and output with the number of units equal to the output shape
        self.edges[self.edge_id] = Edge(self.edge_id, self.vertices[2], self.vertices[1], units=self.output_shape)
        self.edge_id += 1

        # link the input and output (switch on mutability temporarily)
        # switch on mutability
        self.vertices[1].mutable_in = True

        # add the link between the input and flatten vertices
        self.vertices[0].add_edge_out(self.edges[0])
        self.vertices[2].add_edge_in(self.edges[0])

        # add the link between the flatten and output vertices
        self.vertices[2].add_edge_out(self.edges[1])
        self.vertices[1].add_edge_in(self.edges[1])

        # switch back off mutability
        self.vertices[1].mutable_in = False