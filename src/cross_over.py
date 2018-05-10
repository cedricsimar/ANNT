
from dna import DNA

from collections import deque

class Cross_Over:

    def __init__(self, parent_1, parent_2):

        self.parent_1 = DNA()
        self.parent_2 = DNA()


    def breed(self):



        return(" baby ")


    def bridges(self, dna = DNA()):
        
        """
        Stupid algorithm to return all the bridges (vertices which, if deleted cut 
        the graph in two non-connected subgraph) of a NN graph from DNA
        """

        bridges = []

        list_of_vertices = list(dna.vertices.keys())
        list_of_vertices.remove(0)
        list_of_vertices.remove(1)
        
        for bridge_candidate in list_of_vertices:

    
    def is_bridge(self, v_id, dna):

        is_a_bridge = False
        queue = deque([])

        start_vertex = dna.vertices[0]
        output_id = 1 # objective to reach
        
        for out_vertices in start_vertex.


        


        return(is_a_bridge)

        




