import numpy as np





class Layer:
    last_a = None
    last_z = None
    cells: list



    def __init__(self, x: list, y, activation: str, bias = None,weights = None):
        self.x = x
        self.y = y
        self.numberCells = len(x)
        if weights is None:
            self.__initweights()
        else:
            self.weights = weights
        self.bias = bias
        self.cells = np.zeros(self.numberCells)
        if activation == 'relu' or activation == 'sigmoid' or activation == 'none':
            self.activation = activation
        else:
            print('Bitte richtige Aktivierung angeben')

    def __initweights(self):
        return

    def getnumbercells(self) -> int:
        return self.numberCells