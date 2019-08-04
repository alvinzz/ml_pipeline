from parameter import Parameter

class Model(Parameter):
    def __init__(self):
        super().__init__()

    def init_parameters(self):
        print("TODO: create self.params dict for Model")
        raise NotImplementedError

    def predict(self, input):
        print("TODO: define predict method for Model")
        raise NotImplementedError