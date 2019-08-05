from parameter import Parameter

class Evaluator(Parameter):
    param_type = 'Evaluator'

    def __init__(self):
        self.val_data_loc = None
        self.test_data_loc = None

    def update_parameters(self):
        self.params = {
            'param_type': Evaluator.param_type,
        
            'val_data_loc': self.val_data_loc,
            'test_data_loc': self.test_data_loc,
        }

    def load_val_data(self):
        if not hasattr(self, val_data):
            print("TODO: implement load_val_data method for Evaluator")
            raise NotImplementedError

    def load_test_data(self):
        if not hasattr(self, test_data):
            print("TODO: implement load_test_data method for Evaluator")
            raise NotImplementedError

    def val(self, model, log_dir):
        self.load_val_data()
        print("TODO: implement val method for Evaluator")
        raise NotImplementedError

    def test(self, model, log_dir):
        self.load_test_data()
        print("TODO: implement test method for Evaluator")
        raise NotImplementedError