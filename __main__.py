import MLP as mlp
import Layer as lay

def __main__():
    mlp.MLP(input=4)

    mlp.MLP.add(lay.Layer(x=[1, 2, 3], y=[1], activation='none'))
    return