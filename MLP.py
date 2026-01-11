from h5py.h5pl import append

import Layer as lay

class MLP:
    #layerlist: list[lay.Layer]

    def __init__(self, layerlist: list[lay.Layer] = None, input: int = 0):
        self.layerlist = layerlist
        self.input = input

    def add(self, layer: lay.Layer):
        if self.layerlist is None:
            self.layerlist = list[layer]
        else:
            numberCells = self.layerlist.__getitem__(len(self.layerlist) -1).getnumbercells()





