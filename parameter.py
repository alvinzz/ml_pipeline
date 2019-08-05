import json
import glob
import datetime

class Parameter(object):
    def __init__(self):
        pass

    def update_parameters(self):
        print("TODO: update self.params dict {param: value}")
        raise NotImplementedError

    def save(self, name):
        self.update_parameters()
        params = self.save_dict()
        currentDT = datetime.datetime.now()
        name += '_' + currentDT.strftime("%Y_%m_%d_%H_%M_%S") + '.params'
        json.dump(params, open(name, 'w'), sort_keys=False, indent=2)

    def save_dict(self):
        d = {}
        for (param, value) in self.params.items():
            if not isinstance(value, Parameter):
                d[param] = value
            else:
                d[param] = value.save_dict()
        return d

    def load(self, param_file):
        if not glob.glob(param_file):
            matching_pref_files = glob.glob(param_file + '*')
            if not matching_pref_files:
                print("Could not find file with prefix {}".format(param_file))
                raise ValueError
            param_file = sorted(matching_pref_files)[-1]
            print("Could not find {}, instead loading {}")
        params = json.load(open(param_file, 'r'))
        self.load_dict(params)

    def load_dict(self, d):
        for (param, value) in d.items():
            if type(value) != dict:
                setattr(self, param, value)
            else:
                sub_param = getattr(self, param)
                assert isinstance(sub_param, Parameter), "sub-parameter {} of {} has dict of values but is not Parameter".format(sub_param, self)
                sub_param.load_dict(value)

    def print_params(self):
        print(json.dumps(self.params, sort_keys=False, indent=2))