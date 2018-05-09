
from random import random, choice, randint
from settings import Settings
from edge import Edge
from vertex import Vertex

from exceptions import ImmutableException

class Mutation:

    def __init__(self, dna):

        self.dna = dna
        self.mutations={0: self.add_edge, 1: self.remove_edge, 2: self.add_vertex, 3: self.remove_vertex,
                        4: self.flip_edge_type, 5: self.flip_vertex_attribute, 6: self.flip_edge_attribute}

    
    def mutate(self):

        print("Mutating")

        # mutate the dna 
        # for _ in range(self.mutations_per_generation):

        if random() <= Settings.MUTATION_PROBABILITY:

            mutation_keys = list(self.mutations.keys())
            has_mutated = False

            while(not has_mutated and len(mutation_keys) > 0):
                
                # select a random mutation
                mutation = choice(mutation_keys)

                # remove it from the list of possible mutations
                mutation_keys.remove(mutation)

                # try the mutation
                has_mutated = self.mutations[mutation]()

        return (self.dna)
        

    ######################
    # structural mutations

    def add_edge(self):
        has_mutated = False

        # compute the list of vertices with mutable edge_in and edge_out 
        vertices_mutable_out = self.list_of_vertices_mutable_out()
        vertices_mutable_in = self.list_of_vertices_mutable_in()

        # if one of the lists is empty, mutation cannot happen
        if(not len(vertices_mutable_out) or not len(vertices_mutable_in)):
            return (has_mutated)
        
        # choose two vertices to link the edge
        from_v = choice(vertices_mutable_out)
        to_v = choice(vertices_mutable_in)

        # create an edge with a random type which connects the two selected vertices
        edge_type = randint(0, Settings.NUM_EDGE_TYPES - 1)
        new_edge = Edge(self.dna.edge_id, from_v, to_v, type=edge_type)
        self.dna.edge_id += 1

        # update the selected vertices
        try:
            from_v.add_edge_out(new_edge)
            to_v.add_edge_in(new_edge)

        except ImmutableException:
            print("Weird ImmutableException")
            return(has_mutated)


        # since we added a new input edge to a vertex, we check if the vertex action variable
        # is set to SUM or CONCATENATION (because it's the only actions that support multiple
        # input vertices). If it's not the case we choose an action randomly and flip the
        # vertex parameter if possible (properties mutable)
        if to_v.action == Settings.NO_ACTION:
            new_action = randint(Settings.NO_ACTION + 1, Settings.NUM_VERTEX_ACTIONS - 1)
            try:
                to_v.change_action(new_action)

            except ImmutableException:
                print("Weird ImmutableException")
                return(has_mutated)

        # finally mutation can be marked as complete
        has_mutated = True

        return (has_mutated)
    

    def list_of_vertices_mutable_out(self):

        vertices_mutable_out = []
        for vertex_id in self.dna.vertices:
            if(self.dna.vertices[vertex_id].mutable_out):
                vertices_mutable_out.append(self.dna.vertices[vertex_id])

        return (vertices_mutable_out)


    def list_of_vertices_mutable_in(self):

        vertices_mutable_in = []
        for vertex_id in self.dna.vertices:
            if(self.dna.vertices[vertex_id].mutable_in):
                vertices_mutable_in.append(self.dna.vertices[vertex_id])

        return (vertices_mutable_in)


    def list_of_vertices_mutable_prop(self):

        vertices_mutable_prop = []
        for vertex_id in self.dna.vertices:
            if(self.dna.vertices[vertex_id].mutable_properties):
                vertices_mutable_prop.append(self.dna.vertices[vertex_id])

        return (vertices_mutable_prop)
        

    def remove_edge(self):
        has_mutated = False

        return (has_mutated)
        

    def add_vertex(self):
        has_mutated = False

        return (has_mutated)
        

    def remove_vertex(self):
        has_mutated = False

        return (has_mutated)
        


    ######################
    # functional mutations

    def flip_edge_type(self):
        has_mutated = False

        return (has_mutated)
        

    def flip_edge_attribute(self):
        has_mutated = False

        return (has_mutated)
        

    def flip_vertex_attribute(self):
        has_mutated = False

        return (has_mutated)
        
