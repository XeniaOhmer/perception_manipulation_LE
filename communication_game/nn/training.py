import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers


class Trainer:

    def __init__(self, senders, receivers, sender_lr=0.0005, receiver_lr=0.0005, length_cost=0.0,
                 entropy_coeff_sender=0.0, entropy_coeff_receiver=0.0, grad_norm=None, noise_prob=0.):
        
        if isinstance(senders, list):
            self.receivers = receivers
            self.senders = senders
        else: 
            self.senders = [senders]
            self.receivers = [receivers]
        self.n_senders = len(self.senders)
        self.n_receivers = len(self.receivers)   
       
        self.sender_optimizers = [optimizers.Adam(learning_rate=sender_lr) for i in range(self.n_senders)]
        self.receiver_optimizers = [optimizers.Adam(learning_rate=receiver_lr) for i in range(self.n_receivers)]
        self.grad_norm = grad_norm
        self.entropy_coeff_sender = entropy_coeff_sender
        self.entropy_coeff_receiver = entropy_coeff_receiver
        self.length_cost = length_cost
        self.noise_prob = noise_prob
        self.vocab_size = self.senders[0].vocab_size
        self.max_message_length = self.senders[0].max_message_length
        
        if self.n_senders > 1 or self.n_receivers > 1:
            self.population = True 
        else:
            self.population = False

    def train_epoch(self, data_loader, sender_index=None, receiver_index=None):

        sender_loss_epoch = []
        receiver_loss_epoch = []
        rewards_epoch = []
        message_length_epoch = []
        
        if sender_index:
            s_idx = sender_index
        if receiver_index:
            r_idx = receiver_index

        for batch in data_loader:
            
            if not sender_index: 
                s_idx = np.random.randint(self.n_senders)        
            if not receiver_index:
                r_idx = np.random.randint(self.n_receivers)
                
            with tf.GradientTape(persistent=True) as tape:

                tape.watch([self.senders[s_idx].trainable_variables, self.receivers[r_idx].trainable_variables])
                
                sender_input, receiver_input, labels = batch
                message, sender_logits, entropy_sender, message_length, _ = self.senders[s_idx].forward(sender_input)
                
                if self.noise_prob > 0:
                    
                    batch_size = sender_input.shape[0]
                    
                    random_indices = np.random.choice([0, 1], 
                                                      size = self.max_message_length * batch_size,
                                                      p = [1-self.noise_prob, self.noise_prob])
                    random_indices = random_indices.reshape((batch_size, self.max_message_length))
                    random_indices[np.sum(random_indices, axis=1)==self.max_message_length,:] = 0
                    
                    random_symbols = (np.random.choice(self.vocab_size-1, size = self.max_message_length * batch_size) + 1)
                    random_symbols = random_symbols.reshape((batch_size, self.max_message_length))
                    
                    message = tf.where(random_indices==1, random_symbols, message)
                
                selection, receiver_logits, entropy_receiver = self.receivers[r_idx].forward(message, receiver_input)
                rewards = tf.reduce_sum(labels * selection, axis=1)

                sender_policy_loss = - tf.reduce_mean(sender_logits * 
                                                      tf.expand_dims(rewards - self.length_cost * message_length, axis=1))
                receiver_policy_loss = - tf.reduce_mean(receiver_logits * selection * tf.expand_dims(rewards, axis=1))

                sender_loss = sender_policy_loss - self.entropy_coeff_sender * tf.reduce_mean(entropy_sender)
                receiver_loss = receiver_policy_loss - self.entropy_coeff_receiver * tf.reduce_mean(entropy_receiver)

            sender_gradients = tape.gradient(sender_loss, self.senders[s_idx].trainable_variables)
            receiver_gradients = tape.gradient(receiver_loss, self.receivers[r_idx].trainable_variables)

            if self.grad_norm is not None:
                sender_gradients = [tf.clip_by_norm(g, self.grad_norm) for g in sender_gradients]
                receiver_gradients = [tf.clip_by_norm(g, self.grad_norm) for g in receiver_gradients]

            self.sender_optimizers[s_idx].apply_gradients(zip(sender_gradients, self.senders[s_idx].trainable_variables))
            self.receiver_optimizers[r_idx].apply_gradients(zip(receiver_gradients, self.receivers[r_idx].trainable_variables))

            rewards_epoch.append(tf.reduce_mean(rewards))
            sender_loss_epoch.append(tf.reduce_mean(sender_loss))
            receiver_loss_epoch.append(tf.reduce_mean(receiver_loss))
            message_length_epoch.append(tf.reduce_mean(message_length))
        
        return np.mean(rewards_epoch), np.mean(message_length_epoch), np.mean(sender_loss_epoch), np.mean(receiver_loss_epoch)
  

    def evaluate(self, data_loader, sender_index=None, receiver_index=None):
                               
        val_rewards = []

        for batch in data_loader:
            
            if not sender_index:
                s_idx = np.random.randint(self.n_senders)           
            else: 
                s_idx = sender_index
            if not receiver_index:
                r_idx = np.random.randint(self.n_receivers)
            else: 
                r_idx = receiver_index
                
            sender_input, receiver_input, labels = batch
            message, _, _, _, _ = self.senders[s_idx].forward(sender_input, training=False)        
            selection, _, _ = self.receivers[r_idx].forward(message, receiver_input, training=False)

            rewards = np.mean(selection == np.argmax(labels, axis=1))
            val_rewards.append(tf.reduce_mean(rewards))
        
        return np.mean(val_rewards)

