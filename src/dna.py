
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


    def __lt__(self, dna2):
        # IDs are unique globally
        return(self.input_vertex_id < dna2.input_vertex_id)
    

    def add_vertex(self, v):

        if(v.id not in self.vertices):
            self.vertices[v.id] = v
            

    def add_edge(self, e):

        if(e.id not in self.edges):
            self.edges[e.id] = e
            

    def remove_vertex(self, v):

        if(v.id in self.vertices):
            self.vertices.pop(v.id)


    def remove_edge(self, e):

        if(e.id in self.edges):
            self.edges.pop(e.id)


    def __str__(self):

        dna_str = ""

        dna_str += "Vertices\n"
        dna_str += "--------"

        for v_id in self.vertices:

            v = self.vertices[v_id]

            dna_str += "\n\nVertex " + str(v.id) + ": "
            if(v.action == 0):
                dna_str += "No action - "
            elif(v.action == 1):
                dna_str += "Sum - "
            elif(v.action == 2):
                dna_str += "Concatenation - "
            
            if(v.activation):
                dna_str += "ReLu - "
            else:
                dna_str += "Linear - "
            
            if(v.max_pooling):
                dna_str += "Max Pooling - "
            
            if(v.dropout):
                dna_str += "Dropout - "
            
            if(v.flatten):
                dna_str += "Flatten"

            dna_str += "\nInput edges : "
            for e_in in v.edges_in:
                dna_str += str(e_in.id) + "  "

            dna_str += "\nOutput edges : "
            for e_out in v.edges_out:
                dna_str += str(e_out.id) + "  "
        
        dna_str += "\n\n"
        dna_str += "Edges\n"
        dna_str += "-----\n\n"

        for e_id in self.edges:

            e = self.edges[e_id]

            dna_str += "Edge " + str(e.id) + " from " + str(e.from_vertex.id) + " to " + str(e.to_vertex.id) + ": "

            if e.type == Settings.FULLY_CONNECTED:
                dna_str += "Fully connected layer\n\n"
            elif e.type == Settings.CONVOLUTIONAL:
                dna_str += "Convolutional layer\n\n"
            elif e.type == Settings.IDENTITY:
                dna_str += "Identity layer\n\n"

        dna_str += "\n\n"
        dna_str += "Input ID: " + str(self.input_vertex_id) + "\n"
        dna_str += "Output ID: " + str(self.output_vertex_id) + "\n"

        return(dna_str)


    def pretty_print(self):

        print("Vertices")
        print("--------", end ='')

        for v_id in self.vertices:

            v = self.vertices[v_id]

            print("\n\nVertex " + str(v.id) + ": ", end='')
            if(v.action == 0):
                print("No action - ", end='')
            elif(v.action == 1):
                print("Sum - ", end='')
            elif(v.action == 1):
                print("Concatenation - ", end='')
            
            if(v.activation):
                print("ReLu - ", end='')
            else:
                print("Linear - ", end='')
            
            if(v.max_pooling):
                print("Max Pooling - ", end='')
            
            if(v.dropout):
                print("Dropout - ", end='')
            
            if(v.flatten):
                print("Flatten", end='')

            print("\nInput edges : ", end='')
            for e_in in v.edges_in:
                print(str(e_in.id) + "  ", end='')

            print("\nOutput edges : ", end='')
            for e_out in v.edges_out:
                print(str(e_out.id) + "  ", end='')

        print("\n\n")
        print("Edges")
        print("-----\n")

        for e_id in self.edges:

            e = self.edges[e_id]

            print("Edge " + str(e.id) + " from " + str(e.from_vertex.id) + " to " + str(e.to_vertex.id) + ": ", end='')

            if e.type == Settings.FULLY_CONNECTED:
                print("Fully connected layer\n")
            elif e.type == Settings.CONVOLUTIONAL:
                print("Convolutional layer\n")
            elif e.type == Settings.IDENTITY:
                print("Identity layer\n")
        
        print("\n")
        

    def create_primitive_structure(self):

        """
        Create a primitive neural network structure that flattens the input and
        uses one dense layer to map the flatten input to the output shape
        The primitive structure should also make possible the evolution toward 
        a more sophisticated topology 

        [input] --- id --- [flatten] --- id --- [buffer] --- fc --- [output]
        """
    
        # create input vertex  
        self.vertices[Settings.GLOBAL_VERTEX_ID] = Vertex(Settings.GLOBAL_VERTEX_ID, mutable=[False, True, False])
        self.input_vertex_id = Settings.GLOBAL_VERTEX_ID
        Settings.GLOBAL_VERTEX_ID += 1

        # create flatten vertex
        self.vertices[Settings.GLOBAL_VERTEX_ID] = Vertex(Settings.GLOBAL_VERTEX_ID, mutable=[True, True, True], flatten=Settings.FLATTEN)
        self.flatten_vertex_id = Settings.GLOBAL_VERTEX_ID
        Settings.GLOBAL_VERTEX_ID += 1

        # create buffer vertex
        self.vertices[Settings.GLOBAL_VERTEX_ID] = Vertex(Settings.GLOBAL_VERTEX_ID, mutable=[True, False, True])
        self.buffer_vertex_id = Settings.GLOBAL_VERTEX_ID
        Settings.GLOBAL_VERTEX_ID += 1

        # create output vertex
        self.vertices[Settings.GLOBAL_VERTEX_ID] = Vertex(Settings.GLOBAL_VERTEX_ID, mutable=[False, False, False])
        self.output_vertex_id = Settings.GLOBAL_VERTEX_ID
        Settings.GLOBAL_VERTEX_ID += 1

        # create an identity edge to connect the input with the flatten vertex
        self.edges[Settings.GLOBAL_EDGE_ID] = Edge(Settings.GLOBAL_EDGE_ID, self.vertices[self.input_vertex_id], self.vertices[self.flatten_vertex_id], type=Settings.IDENTITY)
        input_to_flatten_edge_id = Settings.GLOBAL_EDGE_ID
        Settings.GLOBAL_EDGE_ID += 1

        # create an identity edge to connect the flatten with the buffer vertex
        self.edges[Settings.GLOBAL_EDGE_ID] = Edge(Settings.GLOBAL_EDGE_ID, self.vertices[self.flatten_vertex_id], self.vertices[self.buffer_vertex_id], type=Settings.IDENTITY)
        flatten_to_buffer_edge_id = Settings.GLOBAL_EDGE_ID
        Settings.GLOBAL_EDGE_ID += 1

        # create non-mutable fc edge between the buffer and output vertex with the number of units equal to the output shape
        self.edges[Settings.GLOBAL_EDGE_ID] = Edge(Settings.GLOBAL_EDGE_ID, self.vertices[self.buffer_vertex_id], self.vertices[self.output_vertex_id], units=self.output_shape, mutable=[False, False, False])
        buffer_to_output_edge_id = Settings.GLOBAL_EDGE_ID
        Settings.GLOBAL_EDGE_ID += 1

        # link the vertices and edges of the graph to connect the input with the output
        # switch on mutability temporarily
        self.vertices[self.buffer_vertex_id].mutable_out = True
        self.vertices[self.output_vertex_id].mutable_in = True

        # add the link between the input and flatten vertex
        self.vertices[self.input_vertex_id].add_edge_out(self.edges[input_to_flatten_edge_id])
        self.vertices[self.flatten_vertex_id].add_edge_in(self.edges[input_to_flatten_edge_id])

        # add the link between the flatten and buffer vertex
        self.vertices[self.flatten_vertex_id].add_edge_out(self.edges[flatten_to_buffer_edge_id])
        self.vertices[self.buffer_vertex_id].add_edge_in(self.edges[flatten_to_buffer_edge_id])

        # add the link between the buffer and output vertex
        self.vertices[self.buffer_vertex_id].add_edge_out(self.edges[buffer_to_output_edge_id])
        self.vertices[self.output_vertex_id].add_edge_in(self.edges[buffer_to_output_edge_id])

        # switch mutability back off
        self.vertices[self.buffer_vertex_id].mutable_out = False
        self.vertices[self.output_vertex_id].mutable_in = False