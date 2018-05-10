
from dna import DNA

from collections import deque
from random import choice

class Cross_Over:

    def __init__(self, parent_1, parent_2):

        self.parent_1 = DNA()
        self.parent_2 = DNA()


    def breed(self):

        # compute all the bridges of the parents
        parent_1_bridges = self.bridges(self.parent_1)
        parent_2_bridges = self.bridges(self.parent_2)

        # choose one bridge per parent for the cross-over operation
        bridge_1_vertex = choice(parent_1_bridges)
        bridge_2_vertex = choice(parent_2_bridges)




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

            if(self.is_bridge(bridge_candidate, dna)):
                bridges.append(bridge_candidate)
        
        return(bridges)

    
    def is_bridge(self, v_id, dna):

        is_a_bridge = False
        output_reached = False

        queue = deque([])
        explored = {}

        start_vertex = dna.vertices[0]
        output_id = 1 # objective to reach
        
        # push all out vertices of the root vertex in the queue if not v_id
        for edge_out in start_vertex.edges_out:

            if(edge_out.to_vertex.id != v_id):
                queue.append(edge_out.to_vertex)

        # walk the graph untill reaching the objective
        while(len(queue) > 0 and not output_reached):

            current_vertex = queue.popleft()

            if(current_vertex.id == output_id):
                output_reached = True
            
            else:

                for edge_out in current_vertex.edges_out:

                    if(edge_out.to_vertex.id != v_id and edge_out.to_vertex.id not in explored):
                        queue.append(edge_out.to_vertex)
            
                explored[current_vertex.id] = True


        if not output_reached:
            is_a_bridge = True

        return(is_a_bridge)

        




