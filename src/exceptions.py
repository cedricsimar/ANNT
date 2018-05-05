
class UnknownLayerException(Exception):

    def __init__(self, message = "Unknown layer type"): # self.errors = errors for custom code

        super().__init__(message)


class ImmutableException(Exception):

    def __init__(self, message = "Impossible to change an immutable object"): # self.errors = errors for custom code

        super().__init__(message)