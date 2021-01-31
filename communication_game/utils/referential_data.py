import numpy as np


def make_referential_data(data_set, n_distractors=2, zero_shot_categories=[]):
    """ create sender and receiver training input for referential game, if zero shot is True, call
        make_zero_shot_data before. """
    
    if len(data_set) == 2: 
        targets_sender, target_labels = data_set
        create_targets_receiver = True
    elif len(data_set) == 3: 
        targets_sender, targets_receiver, target_labels = data_set
        create_targets_receiver = False
        
    target_labels_non_hot = np.argmax(target_labels, axis=1)
    
    if len(zero_shot_categories) > 0:
        classical_indices = [i for i in range(len(targets_sender)) 
                             if target_labels_non_hot[i] not in zero_shot_categories]
        targets_sender = targets_sender[classical_indices]
        if not create_targets_receiver: 
            targets_receiver = targets_receiver[classical_indices]
        target_labels = target_labels[classical_indices]
        target_labels_non_hot = target_labels_non_hot[classical_indices]
    
    n_data = len(target_labels)
    n_total = max(target_labels_non_hot) + 1
    classes = [c for c in range(n_total) if c not in zero_shot_categories]

    distractors = [np.zeros_like(targets_sender) for _ in range(n_distractors)]
    distractor_labels = [np.zeros_like(target_labels) for _ in range(n_distractors)]
    
    possible_distractor_indices = [np.where(target_labels_non_hot != i)[0] for i in classes]
    
    if create_targets_receiver:
        targets_receiver = np.zeros_like(targets_sender)
        for cat in classes:
            targets_receiver_cat = targets_sender[target_labels_non_hot == cat]
            np.random.shuffle(targets_receiver_cat)
            targets_receiver[target_labels_non_hot == cat] = targets_receiver_cat

    for j, l in enumerate(classes):
        n_samples = np.sum(target_labels_non_hot == l)
        random_indices = np.random.choice(possible_distractor_indices[j], size=n_samples*n_distractors, replace=False)
        random_indices = np.reshape(random_indices, (n_samples, n_distractors))
        
        for d in range(n_distractors): 
            distractors[d][target_labels_non_hot == l] = targets_sender[random_indices[:, d]]
            distractor_labels[d][target_labels_non_hot == l] = target_labels[random_indices[:, d]]

    receiver_input = np.stack([targets_receiver] + distractors, axis=1)
    referential_labels = np.zeros((len(targets_sender), n_distractors + 1), dtype=np.float32)
    referential_labels[:, 0] = 1

    for i in range(n_data): 
        perm = np.random.permutation(n_distractors + 1)
        receiver_input[i] = receiver_input[i, perm]
        referential_labels[i] = referential_labels[i, perm]

    target_and_distractor_labels = [target_labels] + distractor_labels

    return targets_sender, receiver_input, referential_labels, target_and_distractor_labels


def make_zero_shot_data(sender_data_orig, train_labels_orig, zero_shot_categories=[], receiver_data_orig=None):
    """ this is necessary if zero shot is True: call for training and validation set to remove all
        objects that should be withheld for testing """
    
    train_labels_non_hot = np.argmax(train_labels_orig, axis=1)
    classical_indices = [i for i, label in enumerate(train_labels_non_hot) 
                         if label not in zero_shot_categories]
    zero_shot_indices = [i for i, label in enumerate(train_labels_non_hot) 
                         if label in zero_shot_categories]
    sender_data = sender_data_orig[classical_indices]
    train_labels = train_labels_orig[classical_indices]
    zero_shot_sender = sender_data_orig[zero_shot_indices]
    zero_shot_labels = train_labels_non_hot[zero_shot_indices]
    if receiver_data_orig is not None: 
        receiver_data = receiver_data_orig[classical_indices]
        zero_shot_receiver = receiver_data_orig[zero_shot_indices]
        return (sender_data, receiver_data, train_labels), (zero_shot_sender, zero_shot_receiver, zero_shot_labels)
    else: 
        return (sender_data, train_labels), (zero_shot_sender, zero_shot_labels)


def make_zero_shot_referential_data(dataset, 
                                    zero_shot_dataset, 
                                    n_distractors=2):
    """ given dataset and zero-shot dataset created in make_zero_shot_data, call this function to get the
        referential game input for the zero-shot testing. """

    # get train data without zero shot targets
    
    if len(dataset) == 2: 
        train_data_sender, train_labels = dataset
        zero_shot_sender, zero_shot_labels = zero_shot_dataset
        create_targets_receiver = True
    elif len(dataset) == 3: 
        train_data_sender, train_data_receiver, train_labels = dataset
        zero_shot_sender, zero_shot_receiver, zero_shot_labels = zero_shot_dataset
        create_targets_receiver = False

    n_samples = len(zero_shot_labels)  
    zero_shot_classes = np.unique(zero_shot_labels)
    
    targets_sender = zero_shot_sender
    
    if create_targets_receiver: 
        targets_receiver = np.zeros_like(targets_sender)   
        for cat in zero_shot_classes:
            targets_receiver_cat = targets_sender[zero_shot_labels == cat]
            np.random.shuffle(targets_receiver_cat)
            targets_receiver[zero_shot_labels == cat] = targets_receiver_cat
    else:
        targets_receiver = zero_shot_receiver
        
    distractors = [np.zeros_like(targets_receiver) for _ in range(n_distractors)]
    distractor_labels = [np.zeros_like(zero_shot_labels) for _ in range(n_distractors)]
    
    random_indices = np.random.choice(range(len(train_data_sender)), size=n_samples*n_distractors, replace=False)
    random_indices = np.reshape(random_indices, (n_samples, n_distractors))
    
    for d in range(n_distractors): 
        if create_targets_receiver:
            distractors[d] = train_data_sender[random_indices[:, d]]
        else: 
            distractors[d] = train_data_receiver[random_indices[:, d]]
        distractor_labels[d] = train_labels[random_indices[:, d]]

    receiver_input = np.stack([targets_receiver] + distractors, axis=1)
    referential_labels = np.zeros((len(targets_sender), n_distractors + 1))
    referential_labels[:, 0] = 1

    for i in range(len(targets_receiver)): 
        perm = np.random.permutation(n_distractors + 1)
        receiver_input[i] = receiver_input[i, perm]
        referential_labels[i] = referential_labels[i, perm]

    target_and_distractor_labels = [zero_shot_labels] + distractor_labels
        
    return targets_sender, receiver_input, referential_labels, target_and_distractor_labels, zero_shot_labels







