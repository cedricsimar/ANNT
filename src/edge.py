
from settings import Settings
from exceptions import UnknownLayerException, ImmutableException

class Edge:

    def __init__(self, edge_id, from_vertex, to_vertex, type = Settings.FULLY_CONNECTED, mutable = [True, True, True], units = None):
        
        self.id = edge_id
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex
        self.type = type

        self.units = None
        self.kernels = None
        self.kernel_shape = None
        self.stride = None

        self.mutable_from, self.mutable_to, self.mutable_type = mutable

        # initialize parameters depending on the layer type
        if self.type == Settings.FULLY_CONNECTED:
            
            if units == None:
                self.set_default_fc()
            else:
                # only used to initialize the output FC layer with output shape
                self.units = units 

        elif self.type == Settings.CONVOLUTIONAL:

            self.set_default_conv()
        
        elif self.type == Settings.IDENTITY:
            pass

        else:
            raise UnknownLayerException()

    
    def is_vertex(self):
        return (False)
        
    def set_default_fc(self):
        self.units = Settings.DEFAULT_UNITS

    def set_default_conv(self):
        self.kernels = Settings.DEFAULT_KERNELS
        self.kernel_shape = Settings.DEFAULT_KERNEL_SHAPE
        self.stride = Settings.DEFAULT_STRIDE
    
    def set_type(self, new_type):

        if(self.mutable_type):
            
            if(self.type != new_type):

                self.type = new_type
                if self.type == Settings.FULLY_CONNECTED:
                    self.set_default_fc()
                elif self.type == Settings.CONVOLUTIONAL:
                    self.set_default_conv()
        
        else:
            raise ImmutableException()
