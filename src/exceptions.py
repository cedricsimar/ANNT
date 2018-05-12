
class UnknownLayerException(Exception):

    def __init__(self, message = "Unknown layer type"): # self.errors = errors for custom code

        super().__init__(message)


class ImmutableException(Exception):

    def __init__(self, message = "Impossible to change an immutable object"): # self.errors = errors for custom code

        super().__init__(message)


class InvalidNumberOfEdges(Exception):

    def __init__(self, message = "Invalid number of edges for the selected action"): # self.errors = errors for custom code

        super().__init__(message)


class ImpossibleToBuild(Exception):

    def __init__(self, message = "The Neural Network is impossible to build"): # self.errors = errors for custom code

        super().__init__(message)


class NoBridgeException(Exception):

    def __init__(self, message = "No bridge found in a parent Neural Network"):

        super().__init__(message)