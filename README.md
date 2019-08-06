# ml_pipeline

Pipeline for ML training that records (hyper-)parameters and allows restoring a model using those (hyper-)parameters.

Example usage (requires TensorFlow 2.0)*:
```export PYTHONPATH=$PYTHONPATH:{PATH_TO_ml_pipeline}
cd xor_example
```

For the first run, uncomment the line with 
`XOR_experiment = setup_experiment()`
in run_xor_experiment.py, and comment out the following lines:
`XOR_experiment = Experiment()
XOR_experiment.load("last_params")`

setup_experiment() creates an Experiment with Model, Trainer, and Evaluator attributes, which have futher sub-attributers and hyper-parameters. See run_xor_experiment.py for further details.

Then run `python run_xor_experiment.py`. 
This will create a folder of the format "xor_YYYY_MM_DD_HH_MM_SS". In this folder will be a `params` file, as well as folders for checkpoints (`ckpts`) and TensorBoard logging.

*Install with `pip install tensorflow==2.0.0-beta1` or `pip install tensorflow-gpu==2.0.0-beta1`.
