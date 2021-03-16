# CS6910: Assignment 1
# Authors: EE19S015: Mudit Batra, EE20S008: Vrunda Sukhadia
Scratch code for Feed Forward Neural Network

## Description of Files
1. To implement results along with wandb compilation refer "Assignment_1_final(with wandb)"
2. To replicate results with different configurations refer "Assignment_1_replicate_results.py"
3. To check out our obtained results on colab compiled with wandb refer "Assignment_1_colab_wandb.ipynb"

### Assignment_1_final(with wandb)
Here sweep configuration (.yaml file) is setted up, you all just need to provide wandb login key and if you want change sweep configuration edit this section:
```
sweep_config = {
    'method': 'random', #grid, random, bayes
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values': [5,10]
        },
        'No_hidden_layer': {
            'values': [3,4,5]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },'Neurons':{
            'values': [32,64,128]
        },
        'weight_decay': {
            'values': [0,0.0005,0.5]
        },'mini_batch_size':{
            'values': [1,16,32,64]
        },'loss_optimizer_type':{
            'values': ['vanila','momentum','nag','rmsprop','adam','nadam']
        },'weight_initialization': {
            'values': ['random','xavier']
        },'activation_type':{
            'values': ['sigmoid','tanh','ReLU']
        },

    }
}
```

### Assignment_1_replicate_results.py

If you want to replicate results for particular configuration then no need to go with "Assignment_1_final(with wandb)" file.
We have seperatley defined all the hyperparameter configuration in the "__init__ block" so you can directly change the parameters in the "__init__ block" using below description:
```
  def __init__(self):

    self.No_hidden_layer = 3
    self.No_of_Neurons = [100,100,100]
    self.batch_size = 64
    self.epoch_num = 10
    self.init_type = 'xavier' #random
    self.activation_type = 'ReLU' #sigmoid, tanh
    self.loss_type = 'cross_entropy'
    self.gradient_type = 'nadam' # vanila, momentum, nag, rmsprop, adam, nadam
```
Note: Number of neurons should be kept in a list for eg: [100,100,100], [64,32], etc

### Assignment_1_colab_wandb.ipynb

To check out the results we have obtained in colab refer this file

### Assignment_1_practice_code

Here in this file all practice and raw codes are available

