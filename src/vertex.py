
from settings import Settings
from exceptions import ImmutableException

class Vertex:

    def __init__(self, id, mutable=[True, True, True], action=Settings.NO_ACTION,
                 batch_normalization = Settings.NO_BATCH_NORMALIZATION,
                 activation=Settings.LINEAR, max_pooling=Settings.NO_MAX_POOLING,
                 dropout = Settings.NO_DROPOUT, flatten = Settings.NO_FLATTEN):
        
        # layer -> action -> batch normalization -> activation -> max-pooling -> dropout 
        self.id = id

        self.action = action

        self.batch_normalization = batch_normalization
        self.activation = activation
        self.max_pooling = max_pooling
        self.dropout = dropout
        self.flatten = flatten

        self.edges_in = []
        self.edges_out = []

        self.mutable_in, self.mutable_out, self.mutable_properties = mutable
        
    
    def change_action(self, new_action):
        if self.mutable_properties:
            self.action = new_action
        else:
            raise ImmutableException()


    def add_edge_in(self, edge):
        if self.mutable_in:
            self.edges_in.append(edge)
        else:
            raise ImmutableException()

    def add_edge_out(self, edge):
        if self.mutable_out:
            self.edges_out.append(edge)
        else:
            raise ImmutableException()
           
    def remove_edge_in(self, edge):
        if self.mutable_in:
            self.edges_in.remove(edge)
        else:
            raise ImmutableException()

    def remove_edge_out(self, edge):
        if self.mutable_out:
            self.edges_out.remove(edge)
        else:
            raise ImmutableException()

    def is_vertex(self):
        return(True)
        