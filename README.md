# perception_manipulation_LE

This code accompanies the anonymous submission 
"Manipulating perception in artificial agents to study the effects on language emergence"
to the Annual Meeting of the Cognitive Science Society 2021.

All packages and versions used can be found in 'requirements.txt'.

We use a setup where two agents, a sender and a receiver, play a reference game. The agents have a vision module which 
is extracted from a pretrained CNN, and a language module which is trained during the reference game. Below you find
details about how to do the CNN training, how to train the agents on the communication game, and where to find the 
results and analyses presented in the paper. 


## Data 

We use the 3d shapes data set (Burgess & Kim, 2018), which you can download using

    wget https://storage.cloud.google.com/3d-shapes/3dshapes.h5

or from [here](https://console.cloud.google.com/storage/browser/3d-shapes;tab=objects?prefix=&forceOnObjectsSortingFiltering=false]).


## CNN training 

Provided the file '3dshapes.h5' is in the main directory with the train_cnn.py file, the default training experiments can be run by calling
train_cnn.py without arguments, i.e. by running

python train_cnn.py

in a terminal with the appropriate python packages installed. By default, the script will train a CNN consisting of 2
convolutional layers with 32 channels each and two fully connected layers with 16 nodes each. Models are built using
the GenericNet class in the ./models folder. A CNN with different parameters can be trained by defining a dictionary
following the format in utils.train.load_default_model_params(). The default parameters dictionary is defined as -  

        model_params['conv_depths'] = [32,32]
        model_params['fc_depths'] = [16,16]
        model_params['conv_pool'] = [True, False]

Each entry in 'conv_depths' and 'fc_depths' specifies the channel dimension, with one layer resulting for each entry.
The 'conv_pool' entry specifies whether or not to pool after each convolutional layer. For further customization with
the GenericNet class see 'models/genericnet.py'. Alternatively, any arbitrary keras model can be used by replacing the
'model' variable, provided input and output dimensions are consistent. In order to change the dataset, replace the
load_data() function in 'train_cnn.py' and update the 'input_shape' and 'num_classes' variables. The 'sf_list' variable
defines the smoothing parameters that will be applied. The outer training loop controls which traits will be enforced,
resulting in one training run per trait per smoothing factor.


All trained CNNs that were used in the paper can be found in 'trained_cnns'.

The functions for analyzing the CNN similarities are under 'communication_game/utils/similarity_analysis.py'.


## Communication game 

All the files you need for training the agents are in the folder 'communication_game'. The training script,
'train_referential_game.py', can be run via command line arguments, with all arguments being explained in the code.
The functions that are used for analyzing the emergent language are under 'utils/language_evaluation.py'. They are 
calculated and stored automatically for each run. 

The results for all runs presented in the paper are under 'results/3Dshapes_subset', with the folder 'basic' containing 
the runs where agents have the same bias, and the folder 'mixed' containing the results where a default agent is paired
with a biased agent. For the latter, the first bias always indicates the sender bias, while the second bias always 
indicates the receiver bias. E.g. 'all_default', stands for *all* sender and *default* receiver. A folder for
one run contains the parameters used, the losses and rewards, the results for the metrics presented in the paper as well
as sender and receiver weights at the end of training. 

## Paper results and plots 

The results and plots presented in the paper can be found in the notebooks: 
'communication_game/results/analysis_language_emergence.py' and 'communication_game/results/paper_plots.ipynb'.
The former contains the analyses of the communication game. The latter contains the plots for groundedness and vision
module biases. 








