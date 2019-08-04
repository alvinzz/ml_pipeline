from parameter import Parameter

class Experiment(Parameter):
    def __init__(self):
        super().__init__()

    def init_parameters(self):
        print("TODO: create self.params dict for Experiment (model, trainer, evaluator)")
        raise NotImplementedError
        # self.model = None
        # self.trainer = None
        # self.evaluator = None
        # self.params = {
        #     'model': self.model,
        #     'trainer': self.trainer,
        #     'evaluator': self.evaluator,
        # }

    def train(self, log_file):
        self.trainer.train(self.model, log_file)

    def val(self, log_file):
        self.evaluator.val(self.model, log_file)

    def test(self, log_file):
        self.evaluator.test(self.model, log_file)