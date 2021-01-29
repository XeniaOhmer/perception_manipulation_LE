import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import sys
sys.path.append('/net/store/cogmod/users/xenohmer/PycharmProjects/SimilarityGames/')
from communication_game.utils.referential_data import *
from communication_game.utils.config import *
from communication_game.nn.agents import *
from communication_game.nn.training import *
from communication_game.utils.language_evaluation import *
import os
import pickle
import argparse
import logging
import h5py
from utils.train import load_data

print("os", os.getcwd())

path_prefix = '../'

parser = argparse.ArgumentParser()

parser.add_argument("--activation", type=str, default='tanh')
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--vocab_size", type=int, default=4)
parser.add_argument("--message_length", type=int, default=3)
parser.add_argument("--entropy_coeff_sender", type=float, default=0.02)
parser.add_argument("--entropy_coeff_receiver", type=float, default=0.02)
parser.add_argument("--n_epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--sim_sender",  nargs="*", type=str, default=['default'])
parser.add_argument("--sim_receiver",  nargs="*", type=str, default=['default'])
parser.add_argument("--sf_sender", nargs="*", type=str, default=['0-0'])
parser.add_argument("--sf_receiver", nargs="*", type=str, default=['0-0'])
parser.add_argument("--run", type=str, default='test0')
parser.add_argument("--learning_rate", type=float, default=0.0005)
parser.add_argument("--n_distractors", type=int, default=2)
parser.add_argument("--zero_shot", type=bool, default=False)
parser.add_argument("--mode", type=str, default='basic')
parser.add_argument("--save_every", type=int, default=10)
parser.add_argument("--noise_prob", type=float, default=0.)
parser.add_argument("--train_vision_sender", type=bool, default=False)
parser.add_argument("--train_vision_receiver", type=bool, default=False)
parser.add_argument("--dataset", type=str, default='3Dshapes_subset')
parser.add_argument("--representation_depth", type=int, default=1)
parser.add_argument("--n_shards", type=int, default=2)
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
save_every = args.save_every
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

# load pretrained model

all_cnn_paths, image_dim, n_classes, feature_dims, zero_shot_cats = get_config(dataset)

path = (path_prefix + 'communication_game/results/' + dataset + '/' + str(mode) + '/' + str(run)+'/' +
        'vs' + str(vocab_size) + '_ml' + str(message_length) + '/')

n_senders = len(sim_sender)
n_receivers = len(sim_receiver)
cnn_paths_sender = [all_cnn_paths[sim_sender[i] + sf_sender[i]] for i in range(n_senders)]
cnn_paths_receiver = [all_cnn_paths[sim_receiver[i] + sf_receiver[i]] for i in range(n_receivers)]

print("path prefix", path_prefix + cnn_paths_sender[0])
test = models.load_model(path_prefix + cnn_paths_sender[0])
cnns_sender = [models.load_model(path_prefix + cnn_paths_sender[i]) for i in range(n_senders)]
vision_modules_sender = [tf.keras.Model(inputs=cnns_sender[i].input,
                                        outputs=cnns_sender[i].get_layer(layer_name[representation_depth]).output)
                         for i in range(n_senders)]
                                                         
cnns_receiver = [models.load_model(path_prefix + cnn_paths_receiver[i]) for i in range(n_receivers)]
vision_modules_receiver = [tf.keras.Model(inputs=cnns_receiver[i].input,
                                          outputs=cnns_receiver[i].get_layer(layer_name[representation_depth]).output)
                           for i in range(n_receivers)]

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

# training

all_reward = []
all_length = []
all_sender_loss = []
all_receiver_loss = []
all_val_reward = []

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

if not os.path.exists(path):
    os.makedirs(path)

with open(path + 'params.pkl', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
logging.basicConfig(filename=path + "log_file.txt",
                    level=logging.DEBUG,
                    format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')

if zero_shot:
    zero_shot_categories = zero_shot_cats
    zero_shot_ref_input = []
else:
    zero_shot_categories = []
                                                                 

(train_data, train_labels), (val_data, val_labels), _, _ = load_data((image_dim, image_dim, 3),
                                                                     dataset='3D-shapes',
                                                                     balance_type=2)

print("train data created")

if zero_shot: 
    (train_data, train_labels), _ = make_zero_shot_data(train_data,
                                                        train_labels,
                                                        zero_shot_categories=zero_shot_categories)
    (val_data, val_labels), _ = make_zero_shot_data(val_data,
                                                    val_labels,
                                                    zero_shot_categories=zero_shot_categories)

shard_size = len(train_data) // n_shards
print(shard_size)

print("zero shot and val data created")

import matplotlib.pyplot as plt

for epoch in range(n_epochs):

    for shard in range(n_shards):

        # prepare dataset for referential game
        sender_in, receiver_in, referential_labels, _ = make_referential_data(
            [train_data[shard_size*shard:shard_size*(shard+1)], train_labels[shard_size*shard:shard_size*(shard+1)]],
            n_distractors=n_distractors,
            zero_shot_categories=zero_shot_categories
        )

        plt.imshow(sender_in[0])
        plt.show()
        plt.imshow(receiver_in[0][0])
        plt.show()
        plt.imshow(receiver_in[0][1])
        plt.show()
        plt.imshow(receiver_in[0][2])
        plt.show()

        print('referential train data created ')
        
        train_dataset = tf.data.Dataset.from_tensor_slices((sender_in, receiver_in, referential_labels))

        del sender_in
        del receiver_in
        del referential_labels

        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.batch(batch_size)
        
        rewards, mean_length, sender_loss, receiver_loss = trainer.train_epoch(train_dataset)

        print("epoch trained")

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

        print("referential val data created")

        # evaluate validation accuracy
        val_dataset = val_dataset.batch(batch_size)
        val_reward = trainer.evaluate(val_dataset)
        all_val_reward.append(val_reward)

        del val_dataset
        print("evaluated")

        logging.info("epoch {0}, shard {1}, rewards train {2}, rewards val {3}".format(epoch,
                                                                                       shard,
                                                                                       rewards,
                                                                                       val_reward))

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


# calculate topographic similarities for selected labels
                                                         
for s, sender in enumerate(trainer.senders):
    
    if len(trainer.senders) > 1:
        save_addition= str(s)
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
