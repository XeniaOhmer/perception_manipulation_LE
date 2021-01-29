import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr
import editdistance  # https://pypi.org/project/editdistance/
from communication_game.utils.config import get_attribute_dict, get_feature_information 


def labels_to_attributes(labels, dataset='3Dshapes'):
    
    attribute_dict = get_attribute_dict(dataset)
    
    return [attribute_dict[l] for l in labels]


class CorrelatedPairwiseSimilarity:
    
    def __init__(self):
        pass
    
    @staticmethod
    def compute_similarity(input1, input2, distance_fn1, distance_fn2):
        
        dist1 = distance.pdist(input1, distance_fn1)
        dist2 = distance.pdist(input2, distance_fn2)

        nan_prop1 = np.count_nonzero(np.isnan(dist1)) / len(dist1)
        nan_prop2 = np.count_nonzero(np.isnan(dist2)) / len(dist2)
        if nan_prop1 > 0.05 or nan_prop2 > 0.05:
            topsim = None
        else:
            topsim = spearmanr(dist1, dist2, nan_policy='omit').correlation

        return topsim
    

class TopographicSimilarity(CorrelatedPairwiseSimilarity):
    # calculates the topographic similarity between sender_input and messages
    # topographic similarity a quantitative proxy for compositionality: https://arxiv.org/pdf/2004.09124.pdf
    # (but has also been criticized)
    # basic functionality: calculate the similarities between all input samples, calculate the similarities between all
    #                      messages, and correlate the corresponding similarities
    #                      (essentially very similar to Representation Similarity Analysis)

    def __init__(self):
        super().__init__()
        self.attribute_distance = distance.cosine
        self.message_distance = lambda x, y: editdistance.eval(x, y)
        self.feature_distance = distance.cosine

    def attribute_feature_topsim(self, attributes, features):
        return self.compute_similarity(attributes, features, self.attribute_distance, self.feature_distance)

    def attribute_message_topsim(self, attributes, messages):
        return self.compute_similarity(attributes, messages, self.attribute_distance, self.message_distance)

    def feature_message_topsim(self, features, messages):
        return self.compute_similarity(features, messages, self.feature_distance, self.message_distance)

    def get_all_topsims(self, attributes, features, messages):
        return [self.attribute_feature_topsim(attributes, features),
                self.attribute_message_topsim(attributes, messages),
                self.feature_message_topsim(features, messages)]
    

class RSA(CorrelatedPairwiseSimilarity): 
    
    def __init__(self, sender, receiver, dist=distance.cosine):
        super().__init__()
        self.distance = dist
        self.sender = sender
        self.receiver = receiver
        
    def get_all_RSAs(self, attributes, images):
        
        messages, _, _, _, hidden_sender = self.sender.forward(images, training=False)
        RSA_sender_input = self.compute_similarity(attributes, hidden_sender, self.distance, self.distance)
        
        hidden_receiver = self.receiver.language_module(messages)
        RSA_receiver_input = self.compute_similarity(attributes, hidden_receiver, self.distance, self.distance)
        
        RSA_sender_receiver = self.compute_similarity(hidden_sender, hidden_receiver, self.distance, self.distance)
        
        return RSA_sender_input, RSA_receiver_input, RSA_sender_receiver
    
    def get_all_RSAs_precalc(self, attributes, hidden_sender, messages): 
        
        RSA_sender_input = self.compute_similarity(attributes, hidden_sender, self.distance, self.distance)
        hidden_receiver = self.receiver.language_module(messages)
        RSA_receiver_input = self.compute_similarity(attributes, hidden_receiver, self.distance, self.distance)
        RSA_sender_receiver = self.compute_similarity(hidden_sender, hidden_receiver, self.distance, self.distance)
        return RSA_sender_input, RSA_receiver_input, RSA_sender_receiver
        

class Groundedness:
    # Calculates whether the language is grounded in the attributes of the objects.
    #
    # The measure can be calculated for individual tokens, or for the entire language see:
    # https://people.eecs.berkeley.edu/~nicholas_tomlin/research/papers/ugrad-thesis.pdf
    # In addition we also measure to what degree the language is grounded in what attribute:
    # This is calculated by the frequency with which symbols that are grounded in the different 
    # TYPES of attributes 
    
    def __init__(self, dataset, vocab_size):
        super().__init__()
        self.dataset = dataset
        self.attribute_dict = get_attribute_dict(dataset)
        feature_dims, feature_order = get_feature_information(dataset=dataset)
        self.attribute_order = feature_order
        self.attribute_dims = feature_dims
        self.symbols = list(range(1, vocab_size+1))
        self.vocab_size = vocab_size

    def compute_bos_groundedness(self, messages, attributes):
        """ Bag-of-symbol groundedness.
        
        For each symbol check with which attribute it co-occurs most frequently, 
        and calculate the frequency with which they co-occur. Normalize to get a measure 
        between zero and one (i.e. multiply by the number of dimensions). If a symbol 
        co-occurs equally frequently with mutliple attributes, still calculate the 
        co-occurence frequency but do not store any attribute."""

        message_length = messages.shape[1] 
        grounding_attributes = []
        frequencies = []
        
        for symbol in self.symbols:
            
            # collect the attributes for each instance of co-occurrence
            all_indices_s = []
            for l in range(message_length):
                if len(np.argwhere(messages[:, l] == symbol)) == 1:
                    indices = [int(np.squeeze(np.argwhere(messages[:, l] == symbol)))]
                else: 
                    indices = list(np.squeeze(np.argwhere(messages[:, l] == symbol)))
                all_indices_s += indices
            unique_indices = np.unique(all_indices_s)
            if len(unique_indices) > 0:         
                if len(unique_indices) == 1: 
                    attributes_s = attributes[unique_indices][0]
                else: 
                    attributes_s = np.sum(attributes[unique_indices], axis=0)
                max_attributes_s = np.max(attributes_s)
                all_argmax_values = [i for i, att in enumerate(attributes_s) if att == max_attributes_s]
                grounding_attributes.append(all_argmax_values)
                frequency_s = (max_attributes_s / np.sum(attributes_s)) * len(self.attribute_order)
            else: 
                grounding_attributes.append([np.nan])
                frequency_s = np.nan
            frequencies.append(frequency_s)
                
        base_count = 0
        per_dimension = []
        for att_dim in self.attribute_dims: 
            num_att = len([grounding_attributes[i] for i in range(len(grounding_attributes)) 
                           if base_count <= min(grounding_attributes[i])
                           and max(grounding_attributes[i]) < base_count + att_dim])
            base_count = base_count + att_dim
            per_dimension.append(num_att)
        
        groundings = dict()
        groundings['symbol_attribute'] = grounding_attributes
        groundings['symbol_groundedness'] = frequencies
        
        for idx, att in enumerate(self.attribute_order):
            groundings[att + '_groundedness'] = per_dimension[idx] / np.sum(per_dimension)
        
        groundings['total_groundedness'] = np.nanmean(frequencies)
    
        return groundings

    def compute_pos_groundedness(self, messages, attributes):
        """ Positional groundedness. 
        
        Same as bag-of-symbol groundedness but for each position in the message separately."""
        
        message_length = messages.shape[1]
        grounding_attributes = [[] for _ in range(len(self.symbols))]
        frequencies = [[] for _ in range(len(self.symbols))]
        
        for s, symbol in enumerate(self.symbols): 
            for l in range(message_length):
                if len(np.argwhere(messages[:, l] == symbol)) == 1:
                    indices = [int(np.squeeze(np.argwhere(messages[:, l] == symbol)))]
                else:
                    indices = list(np.squeeze(np.argwhere(messages[:, l] == symbol)))
                if len(indices) > 0:
                    if len(indices) == 1: 
                        attributes_s = attributes[indices][0]
                    else: 
                        attributes_s = np.sum(attributes[indices], axis=0)
                    max_attributes_s = np.max(attributes_s)
                    all_argmax_values = [i for i, att in enumerate(attributes_s) if att == max_attributes_s]
                    grounding_attributes[s].append(all_argmax_values)
                    frequency_s = (max_attributes_s / np.sum(attributes_s)) * len(self.attribute_order)
                else: 
                    frequency_s = np.nan
                    grounding_attributes[s].append([np.nan])
                frequencies[s].append(frequency_s)
                    
        base_count = 0
        per_dimension = np.zeros((len(self.attribute_order), message_length))
        for a, att_dim in enumerate(self.attribute_dims): 
            for l in range(message_length):
                num_att = len([grounding_attributes[i][l] for i in range(len(grounding_attributes)) 
                               if base_count <= min(grounding_attributes[i][l])
                               and max(grounding_attributes[i][l]) < base_count + att_dim])
                per_dimension[a, l] = num_att
            base_count = base_count + att_dim
            
        groundings = dict()
        groundings['symbol_attribute'] = grounding_attributes
        groundings['symbol_groundedness'] = frequencies
        for idx, att in enumerate(self.attribute_order):
            groundings[att + '_groundedness_pos'] = (np.array(per_dimension[idx, :]) /
                                                     np.sum(per_dimension, axis=0))
            groundings[att + '_groundedness'] = np.mean(groundings[att + '_groundedness_pos'])
        groundings['total_groundedness_pos'] = np.nanmean(frequencies, axis=0)
        groundings['total_groundedness'] = np.mean(groundings['total_groundedness_pos'])
        
        return groundings
