
from settings import Settings
from exceptions import UnknownLayerException

class Edge:

    def __init__(self, from_vertex, to_vertex, type = Settings.FULLY_CONNECTED, mutable = [True, True, True], units = None):

        self.from_vertex = from_vertex
        self.to_vertex = to_vertex
        self.type = type

        self.mutable_from, self.mutable_to, self.mutable_type = mutable

        if self.type == Settings.FULLY_CONNECTED:
            
            if units == None:
                self.units = Settings.DEFAULT_UNITS
            else:
                # to initialize the output FC layer with output shape
                self.units = units 

            self.kernels = None
            self.kernel_shape = None
            self.stride = None

        elif self.type == Settings.CONVOLUTIONNAL:

            self.kernels = Settings.DEFAULT_KERNELS
            self.kernel_shape = Settings.DEFAULT_KERNEL_SHAPE
            self.stride = Settings.DEFAULT_STRIDE
            
            self.units = None
        
        else:
            raise UnknownLayerException()
