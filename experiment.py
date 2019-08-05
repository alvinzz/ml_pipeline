from parameter import Parameter

class Experiment(Parameter):
    param_type = 'Experiment'

    def __init__(self):
        self.model = None
        
        self.trainer = None
        self.evaluator = None

    def update_parameters(self):
        self.params = {
            'param_type': Experiment.param_type,
        
            'model': self.model,
        
            'trainer': self.trainer,
            'evaluator': self.evaluator,
        }

    def train(self, log_dir):
        self.trainer.train(self.model, log_dir)

    def val(self, log_dir):
        self.evaluator.val(self.model, log_dir)

    def test(self, log_dir):
        self.evaluator.test(self.model, log_dir)