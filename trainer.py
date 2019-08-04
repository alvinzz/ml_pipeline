from parameter import Parameter

class Trainer(Parameter):
    def __init__(self):
        super().__init__()

    def init_parameters(self):
        print("TODO: create self.params dict for Trainer (data_loc, etc.)")
        raise NotImplementedError
        # self.data_loc = None
        # self.params = {
        #     'data_loc': self.data_loc,
        # }

    def load_data(self):
        if not hasattr(self, data):
            print("TODO: implement load_data method for Trainer")
            raise NotImplementedError

    def train(self, model, log_file):
        self.load_data()
        print("TODO: implement train method for Trainer")
        raise NotImplementedError