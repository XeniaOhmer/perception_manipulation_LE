import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers, Model


class BaseAgent(Model):

    def __init__(self, vocab_size, max_message_length, embed_dim, hidden_dim, vision_module,
                 flexible_message_length, VtoH_activation='linear'):
        super(BaseAgent, self).__init__()
        self.vocab_size = vocab_size
        self.max_message_length = max_message_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.flexible_message_length = flexible_message_length
        self.vision_module = vision_module
        self.vision_module.trainable = False
        self.vision_to_hidden = layers.Dense(hidden_dim, activation=VtoH_activation, name='vision_to_hidden')

    def __call__(self, inputs):
        pass

    def forward(self, *args, **kwargs):
        pass


class Sender(BaseAgent):

    def __init__(self, vocab_size, max_message_length, embed_dim, hidden_dim, vision_module,
                 flexible_message_length=False, activation='linear'):
        super(Sender, self).__init__(vocab_size, max_message_length, embed_dim, hidden_dim,
                                     vision_module, flexible_message_length, VtoH_activation=activation)
        self.language_module = layers.GRUCell(hidden_dim, name='GRU_layer')
        self.hidden_to_output = layers.Dense(vocab_size, activation='linear', name='hidden_to_output')  # must be linear
        if self.max_message_length > 1:
            self.embedding = layers.Embedding(vocab_size, embed_dim, name='embedding')
        self.__build()

    def __build(self):
        input_shape = self.vision_module.input.get_shape().as_list()
        input_shape = [2] + input_shape[1:]
        test_input = tf.zeros(input_shape)
        _ = self.forward(test_input, training=False)

    def forward(self, image, training=True):

        batch_size = tf.shape(image)[0]
        prev_hidden = self.vision_to_hidden(self.vision_module(image))
        cell_input = tf.zeros((batch_size, self.embed_dim))

        sequence = []
        logits = []
        entropy = []
        symbol_mask = []

        for step in range(self.max_message_length):
            h_t, _ = self.language_module(cell_input, [prev_hidden])

            step_logits = tf.nn.log_softmax(self.hidden_to_output(h_t), axis=1)
            step_entropy = -tf.reduce_sum(step_logits * tf.exp(step_logits), axis=1)

            if training:
                symbol = tf.random.categorical(step_logits, 1)
            else:
                symbol = tf.expand_dims(tf.argmax(step_logits, axis=1), axis=1)

            symbol_mask.append(tf.squeeze(tf.cast(symbol == 0, dtype=tf.int32)))

            logits.append(tf.gather_nd(step_logits, symbol, batch_dims=1))
            symbol = tf.squeeze(symbol)
            sequence.append(symbol)
            entropy.append(step_entropy)
            
            if self.max_message_length > 1:
                cell_input = self.embedding(symbol)

        if self.flexible_message_length:
            cumsum = tf.cast(tf.cumsum(tf.stack(symbol_mask, axis=0), axis=0), tf.float32)
            # calculate mask ignoring zeros to determine actual message length
            mask = tf.cast(cumsum == 0, tf.float32)
            message_length = tf.reduce_sum(mask, axis=0)
            # calculate mask including final zero to calculate relevant policies and entropies
            eos_padding = tf.zeros((1, cumsum.shape[1]))
            cumsum = tf.concat([eos_padding, cumsum[:-1, :]], axis=0)
            mask = tf.cast(cumsum == 0, tf.float32)
            message_length_with_zeros = tf.reduce_sum(mask, axis=0)
            sequence = tf.transpose(tf.cast(tf.stack(sequence), tf.float32) * mask, (1, 0))
            logits = tf.transpose(tf.stack(logits) * mask, (1, 0))
            entropy = tf.reduce_sum(tf.transpose(tf.stack(entropy) * mask, (1, 0)), axis=1) / message_length_with_zeros
        else:
            sequence = tf.transpose(tf.cast(tf.stack(sequence), tf.float32), (1, 0))
            logits = tf.transpose(tf.stack(logits), (1, 0))
            entropy = tf.reduce_mean(tf.transpose(tf.stack(entropy), (1, 0)), axis=1)
            message_length = tf.ones_like(entropy) * self.max_message_length

        return sequence, logits, entropy, message_length, h_t
    

class Receiver(BaseAgent):

    def __init__(self, vocab_size, max_message_length, embed_dim, hidden_dim, vision_module,
                 flexible_message_length=False, activation='linear', n_distractors=1, image_dim=32):
        
        super(Receiver, self).__init__(vocab_size, max_message_length, embed_dim, hidden_dim,
                                       vision_module, flexible_message_length, VtoH_activation=activation)
        self.n_distractors = n_distractors
        self.image_dim = image_dim
        # if message length fixed, 0 counts as a standard symbol and no masking is applied
        self.language_module = Sequential([layers.Embedding(vocab_size, embed_dim, name='embedding'),
                                           layers.GRU(hidden_dim, name='GRU_layer')]) # GRU linear by default
        self.__build()

    def __build(self):
        input_shape = self.vision_module.input.get_shape().as_list()
        input_shape = [2, self.n_distractors+1] + input_shape[1:]
        test_input = tf.zeros(input_shape)
        test_messages = tf.zeros((2, self.max_message_length))
        _ = self.forward(test_messages, test_input, training=False)

    def forward(self, message, images, training=True):
        message_embeddings = self.language_module(message)
        image_embeddings = self.vision_to_hidden(
            self.vision_module(tf.reshape(images, (-1, self.image_dim, self.image_dim, 3)))
        )
        image_embeddings = tf.reshape(image_embeddings, (-1, self.n_distractors + 1, self.hidden_dim))
        similarities = tf.reduce_sum(image_embeddings * tf.expand_dims(message_embeddings, axis=1), axis=2)
        logits = tf.nn.log_softmax(similarities, axis=1)
        entropy = - tf.reduce_sum(logits * tf.exp(logits), axis=1)
        if training:
            actions = tf.squeeze(tf.one_hot(tf.random.categorical(logits, 1), depth=self.n_distractors+1))
        else:
            actions = tf.argmax(logits, axis=1)
        return actions, logits, entropy
