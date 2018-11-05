# TfMicro
TfMicro is a micro framework for Tensorflow training similar to Keras but easy and native to Tensorflow. 
Focus only on creating your TF model. All the rest TfMicro will be done for you. 

# Features
* Multiprocessing data template: inherit from tfmicro.Data class to feed your data into model with multiprocessing.
* Callbacks
  * ModelCheckpoint: to save models by monitor value
  * Keyboard stop: to stop training by pressing 'q'
  * Validation/KeyboardValidation: to make validation step when you want
  * ReducingLearningRate/KeyboardLearningRate: change learning rate on the fly by pressing '+'/'-' keys.
  * Testarium: connect Testarium with tfmicro together
  * AccuracyCallback: evaluate accuracy by model.logits & model.labels
  * FafrCallback: evaluate DET curve and EER on epoch end
* Progress bar and info while training
* Custom user indicators and progress bars
* Custom user keyboard operations: bind any key to any actions
* Saving/loading/preloading models and weights
* Tensorboard automatic support
* Production code templates: training and inference are strong separated


# Install
```
git clone git@github.com:makseq/tfmicro.git
cd tfmicro
python setup.py develop
```

# Usage 
See example folder. 
