
from random import random, choice, randint
from copy import deepcopy

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
            print("No candidate vertices to add edge")
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
                print("ImmutableException while changing a vertex action")
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
        

    def remove_edge(self):

        has_mutated = False

        # compute the list of removable edges
        removable_edges = self.list_of_removable_edges()

        # if the candidate list is empty the mutation cannot happen
        if(not len(removable_edges)):
            print("No candidate edge to be removed")
            return(has_mutated)

        edge_to_remove = choice(removable_edges)
        
        # update the vertices to remove the edge
        try:
            edge_to_remove.from_vertex.remove_edge_out(edge_to_remove)
            edge_to_remove.to_vertex.remove_edge_in(edge_to_remove)
        
        except ImmutableException:
                print("Weird ImmutableException during remove_edge")
                return(has_mutated)

        # finally mutation can be marked as complete
        has_mutated = True

        return (has_mutated)


    def list_of_removable_edges(self):

        # only consider as candidate an edge with from_vertex and to_vertex 
        # that have multiple output edges and input edges respectively
        # I guess we don't need to check if the vertex in/out are mutable
        # since they sure are if they have multiple edges in/out

        removable_edges = []
        for edge_id in self.dna.edges:
            if(self.dna.edges[edge_id].from_vertex.num_edges_out() > 1 and self.dna.edges[edge_id].to_vertex.num_edges_in() > 1):
                removable_edges.append(self.dna.edges[edge_id])
        
        return(removable_edges)


    def add_vertex(self):

        # add a vertex after an edge and connect it to the next vertex with an identity edge
        has_mutated = False

        # create the new vertex (not include max pooling and flatten cause it can cause ill-formed nn)
        new_vertex = Vertex(self.dna.vertex_id,
                            activation=choice([Settings.LINEAR, Settings.RELU]),
                            dropout=choice([Settings.NO_DROPOUT, Settings.USE_DROPOUT]))

        self.dna.vertices[self.dna.vertex_id] = new_vertex
        self.dna.vertex_id += 1

        # select possible edges and choose a candidate 
        edges_mutable_to = self.list_of_edges_mutable_to()
        
        # if the candidate list is empty the mutation cannot happen
        if(not len(edges_mutable_to)):
            print("No candidate edge to add a vertex to")
            return (has_mutated)
    
        selected_edge = choice(edges_mutable_to)
        
        # graft the vertex and identity edge in the network

        # create the new identity edge between new vertex and the to_vertex of the selected edge
        new_identity_edge = Edge(self.dna.edge_id, new_vertex,
                                 selected_edge.to_vertex, type=Settings.IDENTITY)

        self.dna.edges[self.dna.edge_id] = new_identity_edge
        self.dna.edge_id += 1

        try:
            selected_edge.to_vertex.add_edge_in(new_identity_edge)

        except ImmutableException:
            print("Weird Immutable exception while connecting identity edge")
            return(has_mutated)

        # cut the connection between the selected edge and the to_vertex and connect the edge
        # to the new vertex

        try:
            selected_edge.to_vertex.remove_edge_in(selected_edge)
            selected_edge.to_vertex = new_vertex

        except ImmutableException:
            print("Weird Immutable exception while grafting the new vertex to the selected edge")
            return(has_mutated)
        
        # finally mutation can be marked as complete
        has_mutated = True

        return (has_mutated)

    
    def list_of_edges_mutable_to(self):

        # select edges that has a mutable to_vertex and that to_vertex has mutable vertex_in

        edges_mutable_to = []
        for edge_id in self.dna.edges:
            if(self.dna.edges[edge_id].mutable_to and self.dna.edges[edge_id].to_vertex.mutable_in):
                edges_mutable_to.append(self.dna.edges[edge_id])

        return (edges_mutable_to)
        

    def remove_vertex(self):
        
        has_mutated = False
        
        # compute the list of removable vertices
        removable_vertices = self.list_of_removable_vertices()

        # if the candidate list is empty the mutation cannot happen
        if(not len(removable_vertices)):
            print("No candidate vertex to be removed")
            return(has_mutated)

        vertex_to_remove = choice(removable_vertices)

        # try to connect edges_in and the to_vertices as one-to-one correspondance
        # if one-to-one correspondance is no longer possible, deal separately with the different scenari 

        index_in = 0
        index_out = 0
        
        while(index_in < len(vertex_to_remove.edges_in) and index_out < len(vertex_to_remove.edges_out)):

            # create a one-to-one connection
            e_out = vertex_to_remove.edges_out[index_out]
            to_v = e_out.to_vertex
            e_in = vertex_to_remove.edges_in[index_in]

            try:
                to_v.remove_edge_in(e_out)
                e_in.to_vertex = to_v
                to_v.add_edge_in(e_in)

                # since we only replace an incoming edge by another one
                # we don't need to check the action parameter of to_v
            
            except ImmutableException:
                print("Weird Immutable exception while removing a vertex during one-to-one connection")
                return (has_mutated)
            
            index_in += 1
            index_out += 1
        
        # handle cases where the number of edges_in is different from the number of edges_out
        if(index_in < len(vertex_to_remove.edges_in)):
            
            # number of edges_in > number of edges_out
            # we connect the remaining edges_in to the last edge_out
            while(index_in < len(vertex_to_remove.edges_in)):
                
                e_in = vertex_to_remove.edges_in[index_in]

                try:
                    e_in.to_vertex = to_v
                    to_v.add_edge_in(e_in)
                        
                    if(len(to_v.edges_in) > 1 and to_v.action == Settings.NO_ACTION):
                        new_action = randint(Settings.NO_ACTION + 1, Settings.NUM_VERTEX_ACTIONS - 1)
                        try:
                            to_v.change_action(new_action)

                        except ImmutableException:
                            print("ImmutableException while changing a vertex action (in while connecting the remaining edges_in to the last edge_out)")
                            return(has_mutated)
                
                except ImmutableException:
                    print("Weird Immutable exception while connecting the remaining edges_in to the last edge_out")
                    return (has_mutated)
                
                index_in += 1

        elif(index_out < len(vertex_to_remove.edges_out)):

            # number of edges_out > number of edges_in
            # we clone the last edge of the last input vertex and connect it to the next vertex
            last_vertex_in = e_in.from_vertex

            while(index_out < len(vertex_to_remove.edges_out)):

                e_out = vertex_to_remove.edges_out[index_out]
                to_v = e_out.to_vertex

                new_edge = deepcopy(e_in)
                new_edge.to_vertex = to_v

                try:
                    to_v.remove_edge_in(e_out)
                    to_v.add_edge_in(new_edge)
                    last_vertex_in.add_edge_out(new_edge)
                
                except ImmutableException:
                    print("Weird Immutable exception while connecting the last vertex to one to_vertex")
                    return (has_mutated)
                
                index_out += 1
            

        # finally mutation can be marked as complete
        has_mutated = True

        return (has_mutated)
        

    def list_of_removable_vertices(self):

        removable_vertices = []

        for vertex_id in self.dna.vertices:

            v = self.dna.vertices[vertex_id]

            # if the vertex has at least one edge_out and it's not the root vertex (index 0)
            if len(v.edges_out) > 0 and vertex_id:

                # add the vertex to the list if all edges_out are connected to vertices
                # that are all mutable_in (otherwise the subsequent reconnection couldn't occur)
                can_add = True
                edge_index = 0
                while(edge_index < len(v.edges_out) and can_add):

                    if(not v.edges_out[edge_index].to_vertex.mutable_in):
                        can_add = False

                    edge_index += 1
                
                if(can_add):
                    removable_vertices.append(v)
        
        return(removable_vertices)

    ######################
    # functional mutations

    def list_of_vertices_mutable_prop(self):

        vertices_mutable_prop = []
        for vertex_id in self.dna.vertices:
            if(self.dna.vertices[vertex_id].mutable_properties):
                vertices_mutable_prop.append(self.dna.vertices[vertex_id])

        return (vertices_mutable_prop)
    

    def list_of_edges_mutable_prop(self):

        edges_mutable_prop = []
        for edge_id in self.dna.edges:
            if(self.dna.edges[edge_id].mutable_type):
                edges_mutable_prop.append(self.dna.edges[edge_id])

        return (edges_mutable_prop)


    def flip_edge_type(self):
        
        has_mutated = False

        edges_mutable_prop = self.list_of_edges_mutable_prop()

        # if the candidate list is empty the mutation cannot happen
        if(not len(edges_mutable_prop)):
            print("No candidate edge to flip type")
            return(has_mutated)

        selected_edge = choice(edges_mutable_prop)
        new_type = choice([Settings.FULLY_CONNECTED, Settings.CONVOLUTIONAL, Settings.IDENTITY])

        try:
            selected_edge.set_type(new_type)
        except ImmutableException:
            print("Weird Immutable exception at flip_edge_type method")
            return (has_mutated)
        
        # finally mutation can be marked as complete
        has_mutated = True

        return (has_mutated)
        

    def flip_edge_attribute(self):
        
        has_mutated = False

        edges_mutable_prop = self.list_of_edges_mutable_prop()

        # if the candidate list is empty the mutation cannot happen
        if(not len(edges_mutable_prop)):
            print("No candidate edge to flip attribute")
            return(has_mutated)

        selected_edge = choice(edges_mutable_prop)

        # in this implementation, mutable properties are limited to 
        #   - the dense units 
        #   - number of kernels

        if selected_edge.type == Settings.FULLY_CONNECTED:
            try:
                selected_edge.set_units(choice(Settings.ALLOWED_UNITS))
            except ImmutableException:
                print("Weird Immutable exception at flip_edge_attribute method")
                return (has_mutated)

        elif selected_edge.type == Settings.CONVOLUTIONAL:
            try:
                selected_edge.set_kernels(choice(Settings.ALLOWED_KERNELS))
            except ImmutableException:
                print("Weird Immutable exception at flip_edge_attribute method")
                return (has_mutated)
            
        # finally mutation can be marked as complete
        has_mutated = True

        return (has_mutated)
        

    def flip_vertex_attribute(self):

        has_mutated = False

        vertices_mutable_prop = self.list_of_vertices_mutable_prop()

        # if the candidate list is empty the mutation cannot happen
        if(not len(vertices_mutable_prop)):
            print("No candidate vertex to flip attribute")
            return(has_mutated)

        selected_vertex = choice(vertices_mutable_prop)

        # in this implementation, mutable properties are limited to 
        #   - 0 action if there are several input edges
        #   - 1 activation
        #   - 2 max_pooling
        #   - 3 dropout
        #   - 4 flatten
        
        selected_property = randint(0, 4)

        # please include native switch/case in Python, pretty please..
        if selected_property == 0:
            if(len(selected_vertex.edges_in) > 1):
                selected_vertex.action = choice([Settings.SUM, Settings.CONCATENATION])
        elif selected_property == 1:
            selected_vertex.activation = choice([Settings.LINEAR, Settings.RELU])
        elif selected_property == 2:
            selected_vertex.max_pooling = choice([Settings.USE_MAX_POOLING, Settings.NO_MAX_POOLING])
        elif selected_property == 3:
            selected_vertex.dropout = choice([Settings.USE_DROPOUT, Settings.NO_DROPOUT])
        elif selected_property == 4:
            selected_vertex.flatten = choice([Settings.FLATTEN, Settings.NO_FLATTEN])

        # finally mutation can be marked as complete
        has_mutated = True
        
        return (has_mutated)
        
