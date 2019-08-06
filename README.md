# ml_pipeline

Pipeline for ML training that records (hyper-)parameters and allows restoring a model using those (hyper-)parameters.

Example usage (requires TensorFlow 2.0)*:
```
export PYTHONPATH=$PYTHONPATH:{PATH_TO_ml_pipeline}
cd xor_example
```

For the first run, uncomment the line with 
```
XOR_experiment = setup_experiment()
```
in run_xor_experiment.py, and comment out the following lines:
```
XOR_experiment = Experiment()
XOR_experiment.load("last_params")
```
`setup_experiment()` creates an Experiment with Model, Trainer, and Evaluator attributes, which have futher sub-attributers and hyper-parameters. See run_xor_experiment.py for further details.

Then run `python run_xor_experiment.py`. This will train the model, and create a folder of the format `xor_YYYY_MM_DD_HH_MM_SS`. In this folder will be a `params` file, as well as folders for checkpoints (`ckpts`) and TensorBoard logging (`train_log`). It will also a save a copy of the `xor_YYYY_MM_DD_HH_MM_SS/params` file to `last_params` for convenience.

Running `tensorboard --logdir xor_YYYY_MM_DD_HH_MM_SS/train_log/`, we find that training is slow. We can pick up where we left off, and try different hyper-parameters as follows.

First, edit `last_params`. Change `trainer/load_checkpoint_dir` to the `"xor_YYYY_MM_DD_HH_MM_SS/"` folder, `trainer/start_epoch` to `50`, and `trainer/optimizer/epsilon` to `1e-7`.

Next, edit `run_xor_experiment.py`. Comment out 
```
XOR_experiment = setup_experiment()
```
, and uncomment
```
XOR_experiment = Experiment()
XOR_experiment.load("last_params")
```
. This creates a model and runs an experiment with the new hyper-parameters.

Now, running `python run_xor_experiment.py` will create a new folder of the format `xor_YYYY_MM_DD_HH_MM_SS`, which contains the parameters, logs, and checkpoints of the new experiment. Running TensorBoard shows that the model has now converged with the new hyper-parameters.

*Install with `pip install tensorflow==2.0.0-beta1` or `pip install tensorflow-gpu==2.0.0-beta1`.
