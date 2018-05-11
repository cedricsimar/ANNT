
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

        self.input_vertex_id = None
        self.flatten_vertex_id = None
        self.output_vertex_id = None

        self.vertices = {}
        self.edges = {}

        self.num_fully_connected_layers = 0 # limit to 2 FC to limit computation time

    
    def add_vertex(self, v):

        if(v.id not in self.vertices):
            self.vertices[v.id] = v
            

    
    def add_edge(self, e):

        if(e.id not in self.edges):
            self.edges[e.id] = e
            
    

    def __str__(self):

        net_str = ""

        # for v in self.vertices.keys():


        for e in self.edges.keys():

            if self.edges[e].type == Settings.FULLY_CONNECTED:
                net_str += "Fully connected layer "
            elif self.edges[e].type == Settings.CONVOLUTIONAL:
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
    
        # create input vertex  
        self.vertices[Settings.GLOBAL_VERTEX_ID] = Vertex(Settings.GLOBAL_VERTEX_ID, mutable=[False, True, False])
        self.input_vertex_id = Settings.GLOBAL_VERTEX_ID
        Settings.GLOBAL_VERTEX_ID += 1

        # create output vertex
        self.vertices[Settings.GLOBAL_VERTEX_ID] = Vertex(Settings.GLOBAL_VERTEX_ID, mutable=[False, False, False])
        self.output_vertex_id = Settings.GLOBAL_VERTEX_ID
        Settings.GLOBAL_VERTEX_ID += 1

        # create flatten vertex
        self.vertices[Settings.GLOBAL_VERTEX_ID] = Vertex(Settings.GLOBAL_VERTEX_ID, mutable=[True, True, True], flatten=Settings.FLATTEN)
        self.flatten_vertex_id = Settings.GLOBAL_VERTEX_ID
        Settings.GLOBAL_VERTEX_ID += 1

        # create an identity edge to connect the input with the flatten vertex
        self.edges[Settings.GLOBAL_EDGE_ID] = Edge(Settings.GLOBAL_EDGE_ID, self.vertices[self.input_vertex_id], self.vertices[self.flatten_vertex_id], type=Settings.IDENTITY)
        input_to_flatten_edge_id = Settings.GLOBAL_EDGE_ID
        Settings.GLOBAL_EDGE_ID += 1

        # create fc edge between the flatten vertex and output with the number of units equal to the output shape
        self.edges[Settings.GLOBAL_EDGE_ID] = Edge(Settings.GLOBAL_EDGE_ID, self.vertices[self.flatten_vertex_id], self.vertices[self.output_vertex_id], units=self.output_shape)
        flatten_to_output_edge_id = Settings.GLOBAL_EDGE_ID
        Settings.GLOBAL_EDGE_ID += 1

        # link the input and output (switch on mutability temporarily)
        # switch on mutability
        self.vertices[self.output_vertex_id].mutable_in = True

        # add the link between the input and flatten vertex
        self.vertices[self.input_vertex_id].add_edge_out(self.edges[input_to_flatten_edge_id])
        self.vertices[self.flatten_vertex_id].add_edge_in(self.edges[input_to_flatten_edge_id])

        # add the link between the flatten and output vertex
        self.vertices[self.flatten_vertex_id].add_edge_out(self.edges[flatten_to_output_edge_id])
        self.vertices[self.output_vertex_id].add_edge_in(self.edges[flatten_to_output_edge_id])

        # switch back off mutability
        self.vertices[1].mutable_in = False