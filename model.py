from parameter import Parameter

class Model(Parameter):
    param_type = 'Model'

    def __init__(self):
        pass

    def update_parameters(self):
        self.params = {
            'param_type': Model.param_type,
        }

    def predict(self, input):
        print("TODO: define predict method for Model")
        raise NotImplementedError