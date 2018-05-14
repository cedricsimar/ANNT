
from dna import DNA
from exceptions import NoBridgeException
from settings import Settings

from collections import deque
from random import choice
from copy import deepcopy

class Cross_Over:

    def __init__(self, parent_1, parent_2):
        
        # deepcopy in case parent 1 and 2 are the same individual
        self.parent_1 = deepcopy(parent_1)
        self.parent_2 = deepcopy(parent_2)


    def breed(self):

        # compute all the bridges of the parents
        parent_1_bridges = self.bridges(self.parent_1)
        parent_2_bridges = self.bridges(self.parent_2)

        if(len(parent_1_bridges) < 1 or len(parent_2_bridges) < 1):
            raise NoBridgeException()

        # choose one bridge per parent for the cross-over operation
        bridge_1_v_id = choice(parent_1_bridges)
        bridge_2_v_id = choice(parent_2_bridges)

        # create the two offsprings from the parents
        offspring_1, offspring_2 = self.create_offsprings(bridge_1_v_id, bridge_2_v_id)

        return(offspring_1, offspring_2)


    def create_offsprings(self, bridge_1_v_id, bridge_2_v_id):
        
        # create the empty offsprings
        offspring_1 = DNA(Settings.INPUT_SHAPE, Settings.OUTPUT_SHAPE)
        offspring_2 = DNA(Settings.INPUT_SHAPE, Settings.OUTPUT_SHAPE)

        # populate the input and output attributes
        offspring_1.input_vertex_id = self.parent_1.input_vertex_id
        offspring_1.output_vertex_id = self.parent_2.output_vertex_id

        offspring_2.input_vertex_id = self.parent_2.input_vertex_id
        offspring_2.output_vertex_id = self.parent_1.output_vertex_id

        # populate the vertices and edges dictionaries until reaching the bridge vertices (included)
        self.grow_offspring(offspring_1, self.parent_1.vertices[self.parent_1.input_vertex_id], bridge_1_v_id)
        self.grow_offspring(offspring_2, self.parent_2.vertices[self.parent_2.input_vertex_id], bridge_2_v_id)

        # swap the edges_out of the two bridges
        swap_tmp = deepcopy(offspring_1.vertices[bridge_1_v_id].edges_out)
        offspring_1.vertices[bridge_1_v_id].edges_out = deepcopy(offspring_2.vertices[bridge_2_v_id].edges_out)
        offspring_2.vertices[bridge_2_v_id].edges_out = swap_tmp

        for e in offspring_1.vertices[bridge_1_v_id].edges_out:
            e.from_vertex = offspring_1.vertices[bridge_1_v_id]

        for e in offspring_2.vertices[bridge_2_v_id].edges_out:
            e.from_vertex = offspring_2.vertices[bridge_2_v_id]

        # swap the mutable_out attribute of the bridges for the case when only one bridge is the buffer
        swap_tmp = offspring_1.vertices[bridge_1_v_id].mutable_out
        offspring_1.vertices[bridge_1_v_id].mutable_out = offspring_2.vertices[bridge_2_v_id].mutable_out
        offspring_2.vertices[bridge_2_v_id].mutable_out = swap_tmp
        
        # populate the rest of the offsprings
        self.grow_offspring(offspring_1, offspring_1.vertices[bridge_1_v_id], -1)
        self.grow_offspring(offspring_2, offspring_2.vertices[bridge_2_v_id], -1)


        # # create a fusion of the two parents
        # parents_fusion = DNA(Settings.INPUT_SHAPE, Settings.OUTPUT_SHAPE)
        # parents_fusion.vertices.update(deepcopy(self.parent_1.vertices))
        # parents_fusion.vertices.update(deepcopy(self.parent_2.vertices))
        # parents_fusion.edges.update(deepcopy(self.parent_1.edges))
        # parents_fusion.edges.update(deepcopy(self.parent_2.edges))

        # # swap the edges_out of the two bridges
        # swap_tmp = deepcopy(parents_fusion.vertices[bridge_1_v_id].edges_out)
        # parents_fusion.vertices[bridge_1_v_id].edges_out = deepcopy(parents_fusion.vertices[bridge_2_v_id].edges_out)
        # parents_fusion.vertices[bridge_2_v_id].edges_out = swap_tmp

        # for e in parents_fusion.vertices[bridge_1_v_id].edges_out:
        #     e.from_vertex = parents_fusion.vertices[bridge_1_v_id]

        # for e in parents_fusion.vertices[bridge_2_v_id].edges_out:
        #     e.from_vertex = parents_fusion.vertices[bridge_2_v_id]

        # # swap the mutable_out attribute of the bridges for the case when only one bridge is the buffer
        # swap_tmp = parents_fusion.vertices[bridge_1_v_id].mutable_out
        # parents_fusion.vertices[bridge_1_v_id].mutable_out = parents_fusion.vertices[bridge_2_v_id].mutable_out
        # parents_fusion.vertices[bridge_2_v_id].mutable_out = swap_tmp

        # # grow the offsprings by walking the two graphs from their respective input
        # offspring_1 = self.grow_offspring(parents_fusion, self.parent_1.input_vertex_id, self.parent_2.output_vertex_id)
        # offspring_2 = self.grow_offspring(parents_fusion, self.parent_2.input_vertex_id, self.parent_1.output_vertex_id)

        print("**************************")
        print("parent 1 : input_id " + str(self.parent_1.input_vertex_id) + " output_id " + str(self.parent_1.output_vertex_id))
        print("parent 2 : input_id " + str(self.parent_2.input_vertex_id) + " output_id " + str(self.parent_2.output_vertex_id))
        print("offspring 1 : input_id " + str(offspring_1.input_vertex_id) + " output_id " + str(offspring_1.output_vertex_id))
        print("offspring 2 : input_id " + str(offspring_2.input_vertex_id) + " output_id " + str(offspring_2.output_vertex_id))
        print("**************************")

        return(offspring_1, offspring_2)


    def grow_offspring(self, offspring, from_vertex, to_vertex_id):

        # create vertex queue
        vertex_q = deque([])

        # start walking the graph from the given vertex
        vertex_q.append(from_vertex)

        while(len(vertex_q) > 0):

            current_vertex = vertex_q.popleft()
            offspring.add_vertex(current_vertex)

            if(current_vertex.id != to_vertex_id):
                    
                for e_out in current_vertex.edges_out:
                    offspring.add_edge(e_out)
                    vertex_q.append(e_out.to_vertex)


    # def grow_offspring(self, parents_fusion, input_vertex_id, output_vertex_id):

    #     # create vertex queue and explored dictionary
    #     vertex_q = deque([])
    #     explored = {}

    #     # create the empty offspring
    #     offspring = DNA(Settings.INPUT_SHAPE, Settings.OUTPUT_SHAPE)

    #     # populate it's vertices and edges attributes
    #     offspring.input_vertex_id = input_vertex_id
    #     offspring.output_vertex_id = output_vertex_id

    #     # start walking the graph from the input vertex
    #     vertex_q.append(parents_fusion.vertices[input_vertex_id])

    #     while(len(vertex_q) > 0):

    #         current_vertex = vertex_q.popleft()
    #         offspring.add_vertex(current_vertex)

    #         for e_out in current_vertex.edges_out:
    #             offspring.add_edge(e_out)
    #             vertex_q.append(e_out.to_vertex)

    #         # band-aid solution in the case when output_vertex_id is not correct
    #         # if(current_vertex.num_edges_out() == 0):
    #         #     offspring.output_vertex_id = current_vertex.id

    #     return(offspring)


    def bridges(self, dna):
        
        """
        Naive algorithm to return all the bridges (vertices which, if deleted cut 
        the graph in two non-connected subgraph) of a NN graph from DNA
        """

        bridges = []

        list_of_vertices = list(dna.vertices.keys())
        list_of_vertices.remove(dna.input_vertex_id)
        try:
            list_of_vertices.remove(dna.output_vertex_id) # output_vertex_id not in dna.vertices because of cross-over?
        except:
            print("======= MAMA ======")
            print(dna.mama)
            print("======= PAPA ======")
            print(dna.papa)
            print("======= BABY ======")
            print(dna)
            exit()
        
        for bridge_candidate in list_of_vertices:

            if(self.is_bridge(bridge_candidate, dna)):
                bridges.append(bridge_candidate)
        
        return(bridges)

    
    def is_bridge(self, v_id, dna):

        is_a_bridge = False
        output_reached = False

        queue = deque([])
        explored = {}

        start_vertex = dna.vertices[dna.input_vertex_id]
        output_id = 1 # objective to reach
        
        queue.append(start_vertex)

        # push all out vertices connected to the root vertex in the queue if not v_id
        # for edge_out in start_vertex.edges_out:

        #     if(edge_out.to_vertex.id != v_id):
        #         queue.append(edge_out.to_vertex)

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

        




