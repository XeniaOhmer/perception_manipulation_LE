from tensorflow.keras import models
from communication_game.utils.referential_data import *
from communication_game.utils.config import *
from communication_game.nn.agents import *
from communication_game.nn.training import *
from communication_game.utils.language_evaluation import *
import os
import pickle
import argparse
import logging
from utils.train import load_data

###################################################
# initialize and save model and training parameters

parser = argparse.ArgumentParser()

parser.add_argument("--activation", type=str, default='tanh',
                    help='activation function of vision to hidden layer')
parser.add_argument("--hidden_dim", type=int, default=128,
                    help="hidden dimension of recurrent layer")
parser.add_argument("--embed_dim", type=int, default=128,
                    help="dimension of embedding layer")
parser.add_argument("--vocab_size", type=int, default=4,
                    help="vocabulary size")
parser.add_argument("--message_length", type=int, default=3,
                    help="maximal messsage length")
parser.add_argument("--entropy_coeff_sender", type=float, default=0.02,
                    help="entropy coefficient for sender, encourages exploration")
parser.add_argument("--entropy_coeff_receiver", type=float, default=0.02,
                    help="entropy coefficient for receiver, encourages exploration")
parser.add_argument("--n_epochs", type=int, default=150,
                    help="number of epochs for training")
parser.add_argument("--batch_size", type=int, default=128,
                    help="batch size")
parser.add_argument("--sim_sender",  nargs="*", type=str, default=['default'],
                    help="sender bias in {'default', 'color', 'shape', 'scale', 'all'}, multiple senders are possible")
parser.add_argument("--sim_receiver",  nargs="*", type=str, default=['default'],
                    help="receiver bias in {'default', 'color', 'shape', 'scale', 'all'}, multiple rec. are possible")
parser.add_argument("--sf_sender", nargs="*", type=str, default=['0-0'],
                    help="smoothing factors for the sender vision modules")
parser.add_argument("--sf_receiver", nargs="*", type=str, default=['0-0'],
                    help="smoothing factors for the receiver vision module")
parser.add_argument("--run", type=str, default='test',
                    help="name of current run, for saving results")
parser.add_argument("--learning_rate", type=float, default=0.0005,
                    help="learning rate")
parser.add_argument("--n_distractors", type=int, default=2,
                    help="number of distractors for the receiver")
parser.add_argument("--zero_shot", type=bool, default=False,
                    help="whether zero shot run or not, if True agents are trained on subset of the data and tested"
                         "on the remaining (novel) items")
parser.add_argument("--mode", type=str, default='basic',
                    help="folder of the current simulation for storing the results")
parser.add_argument("--noise_prob", type=float, default=0.,
                    help="probability that words of the sender are replaced by random other words (not used in paper)")
parser.add_argument("--dataset", type=str, default='3Dshapes_subset',
                    help="which data set to use, only one implemented here")
parser.add_argument("--representation_depth", type=int, default=1,
                    help="CNN layer for vision module output, 0=softmax, 1=penultimate, 2=second to last")
parser.add_argument("--n_shards", type=int, default=2,
                    help="number of shards for the dataset, to be used under RAM constraints")
args = parser.parse_args()

run = args.run
sim_sender = args.sim_sender
sim_receiver = args.sim_receiver
batch_size = args.batch_size
hidden_dim = args.hidden_dim
embed_dim = args.embed_dim
vocab_size = args.vocab_size
message_length = args.message_length
entropy_coeff_sender = args.entropy_coeff_sender
entropy_coeff_receiver = args.entropy_coeff_receiver
learning_rate = args.learning_rate
n_distractors = args.n_distractors
activation = args.activation
n_epochs = args.n_epochs
sf_sender = args.sf_sender
sf_receiver = args.sf_receiver
zero_shot = args.zero_shot
mode = args.mode
noise_prob = args.noise_prob
representation_depth = args.representation_depth
dataset = args.dataset

n_shards = args.n_shards

length_cost = 0
grad_norm = None
entropy_annealing = 1.
layer_name = {0: 'dense_2',
              1: 'dense_1',
              2: 'dense'}

params = {"batch_size": batch_size,
          "similarity_sender": sim_sender,
          "similarity_receiver": sim_receiver,
          "smoothing_factor_sender": sf_sender,
          "smoothing_factor_receiver": sf_receiver,
          "hidden_dim": hidden_dim,
          "embed_dim": embed_dim,
          "vocab_size": vocab_size,
          "message_length": message_length,
          "entropy_coeff_sender": entropy_coeff_sender,
          "entropy_coeff_receiver": entropy_coeff_receiver,
          "length_cost": length_cost,
          "grad_norm": grad_norm,
          "learning_rate": learning_rate,
          "n_distractors": n_distractors,
          "activation": activation,
          "entropy_annealing": entropy_annealing,
          "zero_shot": zero_shot,
          "noise_prob": noise_prob,
          "representation_depth": representation_depth}

path_prefix = '../'
path = (path_prefix + 'communication_game/results/' + dataset + '/' + str(mode) + '/' + str(run)+'/' +
        'vs' + str(vocab_size) + '_ml' + str(message_length) + '/')

if not os.path.exists(path):
    os.makedirs(path)

with open(path + 'params.pkl', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


###################################################
# load pretrained CNNs to be used as vision modules

all_cnn_paths, image_dim, n_classes, feature_dims, zero_shot_cats = get_config(dataset)

n_senders = len(sim_sender)
n_receivers = len(sim_receiver)
cnn_paths_sender = [all_cnn_paths[sim_sender[i] + sf_sender[i]] for i in range(n_senders)]
cnn_paths_receiver = [all_cnn_paths[sim_receiver[i] + sf_receiver[i]] for i in range(n_receivers)]

cnns_sender = [models.load_model(path_prefix + cnn_paths_sender[i]) for i in range(n_senders)]
vision_modules_sender = [tf.keras.Model(inputs=cnns_sender[i].input,
                                        outputs=cnns_sender[i].get_layer(layer_name[representation_depth]).output)
                         for i in range(n_senders)]
                                                         
cnns_receiver = [models.load_model(path_prefix + cnn_paths_receiver[i]) for i in range(n_receivers)]
vision_modules_receiver = [tf.keras.Model(inputs=cnns_receiver[i].input,
                                          outputs=cnns_receiver[i].get_layer(layer_name[representation_depth]).output)
                           for i in range(n_receivers)]


###################################################
# initialize sender, receiver and training classes

senders = [Sender(vocab_size + 1, message_length, embed_dim, hidden_dim, activation=activation,
                  vision_module=vision_modules_sender[i]) for i in range(n_senders)]

receivers = [Receiver(vocab_size + 1, message_length, embed_dim, hidden_dim, activation=activation,
                      vision_module=vision_modules_receiver[i], n_distractors=n_distractors, image_dim=image_dim)
             for i in range(n_receivers)]

trainer = Trainer(senders, receivers, 
                  entropy_coeff_sender=entropy_coeff_sender, 
                  entropy_coeff_receiver=entropy_coeff_receiver, 
                  length_cost=length_cost,
                  grad_norm=grad_norm, 
                  sender_lr=learning_rate,
                  receiver_lr=learning_rate, 
                  noise_prob=noise_prob)


###################################################
# training

all_reward = []
all_length = []
all_sender_loss = []
all_receiver_loss = []
all_val_reward = []
    
logging.basicConfig(filename=path + "log_file.txt",
                    level=logging.DEBUG,
                    format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')

if zero_shot:
    zero_shot_categories = zero_shot_cats
    zero_shot_ref_input = []
else:
    zero_shot_categories = []
                                                                 
# load data
(train_data, train_labels), (val_data, val_labels), _, _ = load_data((image_dim, image_dim, 3),
                                                                     dataset='3D-shapes',
                                                                     balance_type=2)

if zero_shot: 
    (train_data, train_labels), _ = make_zero_shot_data(train_data,
                                                        train_labels,
                                                        zero_shot_categories=zero_shot_categories)
    (val_data, val_labels), _ = make_zero_shot_data(val_data,
                                                    val_labels,
                                                    zero_shot_categories=zero_shot_categories)

shard_size = len(train_data) // n_shards

# each epoch train and valdidate the agents
for epoch in range(n_epochs):

    for shard in range(n_shards):

        # prepare dataset for referential game
        sender_in, receiver_in, referential_labels, _ = make_referential_data(
            [train_data[shard_size*shard:shard_size*(shard+1)], train_labels[shard_size*shard:shard_size*(shard+1)]],
            n_distractors=n_distractors,
            zero_shot_categories=zero_shot_categories
        )
        
        train_dataset = tf.data.Dataset.from_tensor_slices((sender_in, receiver_in, referential_labels))

        del sender_in
        del receiver_in
        del referential_labels

        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.batch(batch_size)
        
        rewards, mean_length, sender_loss, receiver_loss = trainer.train_epoch(train_dataset)

        trainer.entropy_coeff_receiver = trainer.entropy_coeff_receiver * entropy_annealing
        trainer.entropy_coeff_sender = trainer.entropy_coeff_sender * entropy_annealing
        
        all_reward.append(rewards)
        all_length.append(mean_length)
        all_sender_loss.append(sender_loss)
        all_receiver_loss.append(receiver_loss)

        del train_dataset

        # get validation data and messages
        sender_in, receiver_in, referential_labels, _ = make_referential_data([val_data, val_labels],
                                                                              n_distractors=n_distractors,
                                                                              zero_shot_categories=zero_shot_categories)
        val_dataset = tf.data.Dataset.from_tensor_slices((sender_in, receiver_in, referential_labels))
        del sender_in
        del receiver_in
        del referential_labels

        # evaluate validation accuracy
        val_dataset = val_dataset.batch(batch_size)
        val_reward = trainer.evaluate(val_dataset)
        all_val_reward.append(val_reward)
        del val_dataset

        logging.info("epoch {0}, shard {1}, rewards train {2}, rewards val {3}".format(epoch,
                                                                                       shard,
                                                                                       rewards,
                                                                                       val_reward))
    # store final agent models
    if epoch == n_epochs-1:
        for r, receiver in enumerate(trainer.receivers):
            if len(trainer.receivers) > 0:
                receiver.save_weights(path + 'receiver' + str(r) + '_weights_epoch' + str(epoch) + '/')
            else: 
                receiver.save_weights(path + 'receiver_weights_epoch' + str(epoch) + '/')
        for s, sender in enumerate(trainer.senders):
            if len(trainer.senders) > 0: 
                sender.save_weights(path + 'sender' + str(s) + '_weights_epoch' + str(epoch) + '/')
            else: 
                sender.save_weights(path + 'sender_weights_epoch' + str(epoch) + '/')

np.save(path + 'reward.npy', all_reward)
np.save(path + 'length.npy', all_length)
np.save(path + 'sender_loss.npy', all_sender_loss)
np.save(path + 'receiver_loss.npy', all_receiver_loss)
np.save(path + 'val_reward.npy', all_val_reward)


#############################################################
# evaluate language with different metrics and store results

# get validation data and messages
(train_data, train_labels), (val_data, val_labels), _, _ = load_data((image_dim, image_dim, 3),
                                                                     dataset='3D-shapes', balance_type=2)
print("data for language analysis loaded")

if zero_shot:
    (train_data, train_labels), (zs_targets, zs_labels) = make_zero_shot_data(
        train_data, train_labels, zero_shot_categories=zero_shot_categories)

sender_in, receiver_in, referential_labels, target_distractor_labels = make_referential_data(
    [val_data, val_labels], n_distractors=n_distractors, zero_shot_categories=[])
target_labels = np.argmax(target_distractor_labels[0], axis=1)
target_attributes = np.array(labels_to_attributes(target_labels, dataset=dataset))

# evaluation topsim & messages

# select 50 samples for each label (reduces computation time)
selected_indices = []
for i in range(n_classes):
    if i not in zero_shot_categories:
        indices = np.where(target_labels == i)[0][0:50]
        selected_indices += [ind for ind in indices]
selected_indices = np.array(selected_indices)


# calculate topographic similarities, groundedness, and RSA  for selected labels
for s, sender in enumerate(trainer.senders):
    
    if len(trainer.senders) > 1:
        save_addition = str(s)
    else:
        save_addition = ''
    
    target_features = sender.vision_module(sender_in)
    messages, logits, entropy, message_length, hidden_sender = sender.forward(sender_in, training=False)
    
    rsa = RSA(sender, trainer.receivers[s])
    RSA_sender_input, RSA_receiver_input, RSA_sender_receiver = rsa.get_all_RSAs_precalc(
        target_attributes[selected_indices], 
        hidden_sender.numpy()[selected_indices], 
        messages.numpy()[selected_indices])
             
    groundedness = Groundedness(dataset, vocab_size)
    posground = groundedness.compute_pos_groundedness(messages.numpy()[selected_indices], 
                                                      target_attributes[selected_indices])
    bosground = groundedness.compute_bos_groundedness(messages.numpy()[selected_indices], 
                                                      target_attributes[selected_indices])
    
    topsim = TopographicSimilarity()
    similarities = topsim.get_all_topsims(target_attributes[selected_indices],
                                          list(target_features.numpy()[selected_indices]),
                                          list(messages.numpy()[selected_indices]))

    # save all in evaluation dict
    eval_dict = {'message_length': np.mean(message_length),
                 'entropy': np.mean(entropy),
                 'topsim_attributes_features': similarities[0],
                 'topsim_attributes_messages': similarities[1],
                 'topsim_features_messages': similarities[2],
                 'rsa_sender_input': RSA_sender_input,
                 'rsa_receiver_input': RSA_receiver_input,
                 'rsa_sender_receiver': RSA_sender_receiver, 
                 'posground': posground, 
                 'bosground': bosground
                 }

    # evaluation zero-shot
    if zero_shot:
        zero_shot_test = make_zero_shot_referential_data((train_data, train_labels),
                                                         (zs_targets, zs_labels),
                                                         n_distractors=n_distractors)
        if n_distractors == 2: 
            zero_shot_test = pickle.load(open(
                path_prefix + 'communication_game/' + dataset + '_shards/zero_shot_referential_data.pkl','rb'))
        
        if len(zero_shot_test[0]) % batch_size == 1:
            zero_shot_test[0] = zero_shot_test[0][:-1]
            zero_shot_test[1] = zero_shot_test[1][:-1]
            zero_shot_test[2] = zero_shot_test[2][:-1]

        zero_shot_dataset = tf.data.Dataset.from_tensor_slices((zero_shot_test[0],
                                                                zero_shot_test[1],
                                                                zero_shot_test[2]))
        del zero_shot_test
        
        zero_shot_dataset = zero_shot_dataset.batch(batch_size)
        zero_shot_reward = trainer.evaluate(zero_shot_dataset)
        
        del zero_shot_dataset
        eval_dict['zero_shot_reward'] = zero_shot_reward
    
    with open(path + 'eval' + save_addition + '.pkl', 'wb') as handle:
        pickle.dump(eval_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # evaluation messages
    message_dict = {}
    sender_out, _, _, _, _ = sender.forward(sender_in, training=False)
    indices = []
    for i in range(n_classes):
        for l, label in enumerate(target_labels):
            if label == i:
                indices.append(l)
                break
    
    if len(feature_dims) == 2:
        for i in range(feature_dims[0]):
            for j in range(feature_dims[1]):
                label = i * feature_dims[1] + j
                message_dict['hue' + str(i) + '_shape' + str(j)] = np.array(sender_out[indices[label]])[:]
                
    elif len(feature_dims) == 3:
        for i in range(feature_dims[0]):
            for j in range(feature_dims[1]):
                for k in range(feature_dims[2]):
                    label = i * feature_dims[1] + j*feature_dims[2] + k
                    message_dict['hue' + str(i) + '_scale' + str(j) + '_shape' + str(k)] = np.array(
                        sender_out[indices[label]])[:]

    with open(path + 'message_dict' + save_addition + '.pkl', 'wb') as handle:
        pickle.dump(message_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
