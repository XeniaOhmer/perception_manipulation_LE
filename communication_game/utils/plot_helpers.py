import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def get_stats(name, vocab_size, message_length, path, mode, n_vision, n_runs=1):
    
    eval_dicts = []
    message_dicts = []
    rewards = []
    val_rewards = []

    for run in range(n_runs): 
        
        try: 
            if n_runs > 1: 
                loadpath = path + mode + '/' + name + str(run) + '/vs' + str(vocab_size) + '_ml' + str(message_length)
            else: 
                loadpath = path + mode + '/' + name + '/vs' + str(vocab_size) + '_ml' + str(message_length)
            rewards.append(np.load(loadpath + '/reward.npy'))
            val_rewards.append(np.load(loadpath + '/val_reward.npy'))
            for i in range(n_vision):
                if n_vision > 1: 
                    append = str(i)
                else: 
                    append = ''
                eval_dicts.append(pickle.load(open(loadpath + '/eval' + append + '.pkl', 'rb')))
                message_dicts.append(pickle.load(open(loadpath + '/message_dict' + append + '.pkl', 'rb')))
        except: 
            print('not included', name, run)
            continue

    return rewards, val_rewards, eval_dicts, message_dicts


def groundedness_dataframe(names, vocab_size, message_length, path='./', n_vision=1, n_runs=1):

    run_names = []
    features = []
    groundedness = []
    relevant_keys = ['color_groundedness', 'scale_groundedness', 'shape_groundedness']

    for name_idx, name in enumerate(names):  
        
        if '_' in name:
            mode = 'mixed'
        else: 
            mode = 'basic'

        _, _, evals, _ = get_stats(name, vocab_size, message_length, path, mode, n_vision, n_runs=n_runs)

        for key_idx, key in enumerate(relevant_keys): 

            for run in range(n_runs):

                if mode == 'mixed':
                    both_agents = name.split('_')
                    run_name = [both_agents[i] for i in range(2) if both_agents[i]!='default'][0]
                    if run_name == 'scale':
                        run_name = 'size'
                    run_names.append(run_name)
                else: 
                    if name == 'scale':
                        run_name = 'size'
                    else: 
                        run_name = name
                    run_names.append(run_name)
                f = key[0:-13]
                if f == 'scale':
                    f = 'size'
                features.append(f)
                groundedness.append(evals[run]['bosground'][key])

    d = {'bias': run_names, 'feature': features, 'G_f': groundedness}
    df = pd.DataFrame(data=d)
    return df


def show_results_multiples(names, vocab_size, message_length, ylim=(0.8, 1.01), path='./', mode='basic', n_vision=1,
                           subplots=(1, 4), n_runs=5, n_shards=2, figsize=(15, 3.5)):
    
    fig = plt.figure(figsize=figsize)
    for plot_index, name in enumerate(names): 
                 
        plt.subplot(subplots[0], subplots[1], plot_index+1)
        
        rewards, val_rewards, evals, messages = get_stats(name, vocab_size, message_length, path, mode, n_vision,
                                                          n_runs=n_runs)
        
        mean_R_train = np.mean([R[0::n_shards] for R in rewards], axis=0)
        max_R_train = np.max([R[0::n_shards] for R in rewards], axis=0)
        min_R_train = np.min([R[0::n_shards] for R in rewards], axis=0)
        
        plt.plot(mean_R_train, color='k')
        plt.plot(min_R_train, color='b', alpha=0.3)
        plt.plot(max_R_train, color='r', alpha=0.3)
        plt.fill_between(range(len(min_R_train)), min_R_train, y2=mean_R_train, color='blue', alpha=.1)
        plt.fill_between(range(len(max_R_train)), mean_R_train, y2=max_R_train, color='red', alpha=.1)
        
        plt.ylim(ylim)
        
        final_R_train = mean_R_train[-1]
        final_R_val = np.mean([R for R in val_rewards], axis=0)[-1]

        topsim = np.mean([evals[i]['topsim_attributes_messages'] for i in range(len(evals))])   
        if 'zero_shot' in name or 'zs' in name:  
            zero_shot = np.mean([evals[i]['zero_shot_reward'] for i in range(len(evals))])
            zs = True
        else:
            zs = False

        if '/' in name: 
            name = name.partition('/')[0]

        title = str(name) + '\ntrain: ' + str(round(final_R_train, 3)) + ', val: ' + str(round(final_R_val, 3))

        if message_length > 1:
            title = title + '\ntopsim: ' + str(round(topsim, 3))
        if zs: 
            if message_length > 1:
                title = title + ', R zs: ' + str(round(zero_shot, 3))
            else: 
                title = title + '\nR zs: ' + str(round(zero_shot, 3))

        plt.title(title)
        plt.xlabel('epoch')
        plt.ylabel('reward')
        
    fig.legend(labels=['mean', 'min', 'max'],   # The labels for each line
               loc="lower center",   # Position of legend
               borderaxespad=-0.3,    # Small spacing around legend box
               ncol=n_runs)
               
    fig.tight_layout()
    

def show_accuracies(names, vocab_size, message_length, path='./', mode='basic', n_runs=5, n_vision=1):
    
    all_zero_shots = []
    for name in names: 
        
        print('\n'+name)
        
        rewards, val_rewards, evals, messages = get_stats(name, vocab_size, message_length, path, mode, n_vision,
                                                          n_runs=n_runs)
        
        mean_R_train = np.mean([R[-1] for R in rewards])
        mean_R_val = np.mean([R[-1] for R in val_rewards])
        std_R_train = np.std([R[-1] for R in rewards])
        std_R_val = np.std([R[-1] for R in val_rewards])
        
        print('train reward: ', round(mean_R_train, 3), '+-', round(std_R_train, 3))
        print('val reward: ', round(mean_R_val, 3), '+-', round(std_R_val, 3))
        
        if 'zero_shot' in name or 'zs' in name:  
            mean_zs = np.mean([evals[i]['zero_shot_reward'] for i in range(len(evals))])
            std_zs = np.std([evals[i]['zero_shot_reward'] for i in range(len(evals))])
            print('zero shot', round(mean_zs, 3), '+-', round(std_zs, 3))
            all_zero_shots.append([evals[i]['zero_shot_reward'] for i in range(len(evals))])
        
    if len(all_zero_shots) > 0: 
        print(ttest_ind(all_zero_shots[0], all_zero_shots[-1]))
            
            
def show_results(names, vocab_size, message_length, ylim=(0.8, 1.01), path='./', mode='basic',
                 n_vision=1, subplots=(1, 4)):
    
    for plot_index, name in enumerate(names): 
                 
        plt.subplot(subplots[0], subplots[1], plot_index+1)
    
        rewards, val_rewards, evals, messages = get_stats(name, vocab_size, message_length, path, mode, n_vision)
        
        for i in range(len(rewards)): 
            R_train = rewards[i]
            plt.plot(R_train, alpha=0.5)  
            if len(val_rewards) > 0:
                R_val = val_rewards[i]
                plt.plot(R_val, alpha=0.5)
        plt.ylim(ylim)
        plt.legend(['train', 'val'])

        final_R_train = R_train[-1]
        if len(val_rewards) > 0:
            final_R_val = R_val[-1]
        else:
            final_R_val = evals[0]['val_reward']

        topsim = np.mean([evals[i]['topsim_attributes_messages'] for i in range(len(evals))])   
        if 'zero_shot' in name or 'zs' in name: 
            zero_shot = np.mean([evals[i]['zero_shot_reward'] for i in range(len(evals))])
            zs = True
        else:
            zs = False
        
        if '/' in name: 
            name= name.partition('/')[0]
            
        title = str(name) + '\nR train: ' + str(round(final_R_train,3)) + ', R val: ' + str(round(final_R_val, 3))
        
        if message_length > 1:
            title = title + '\ntopsim: ' + str(round(topsim, 3))
        if zs: 
            if message_length > 1:
                title = title + ', R zs: ' + str(round(zero_shot, 3))
            else: 
                title = title + '\nR zs: ' + str(round(zero_shot, 3))

        plt.title(title)

        plt.xlabel('epoch')
        plt.ylabel('reward')
    
    
def show_messages(names, 
                  vocab_size, 
                  message_length, 
                  path='./',
                  mode='basic', 
                  n_vision=1,
                  dataset='3Dshapes'):
    
    for name in names: 
        
        _, _, _, messages = get_stats(name, vocab_size, message_length, path, mode, n_vision)
        
        if '/' in name: 
            name = name.partition('/')[0]
        print(name)
        
        if dataset == '3Dshapes':
            for i, m in enumerate(messages):
                if len(messages) > 1:
                    print(i)
                m_table = np.zeros((10,4,message_length))
                for _ in m.keys():
                    for hue in range(10):
                        for shape in range(4):
                            m_table[hue, shape, :] = m['hue'+str(hue)+'_shape'+str(shape)]
                m_table[m_table == 10] = 0
                for line in m_table:
                    print(*line)
                    
        if dataset == '3Dshapes_subset':
            for i, m in enumerate(messages):
                if len(messages) > 1:
                    print(i)
                m_table = np.zeros((4, 4, 4, message_length))
                for _ in m.keys():
                    for hue in range(4):
                        for shape in range(4):
                            for size in range(4): 
                                m_table[hue, shape, size, :] = m['hue' + str(hue) +
                                                                 '_shape' + str(shape) +
                                                                 '_size' + str(size)]
                m_table[m_table == 10] = 0
                for hue in range(4):
                    print('hue', str(hue+1))
                    for line in m_table[hue, :, :, :]:
                        print(*line)


def show_topsims(names, vocab_size, message_length, path='./', mode='basic', n_vision=1, n_runs=1):
    
    all_topsims_am = []
    
    print('topsims')
    for name in names:
    
        _, _, evals, _ = get_stats(name, vocab_size, message_length, path, mode, n_vision, n_runs=n_runs)

        topsim_am = round(np.mean([evals[i]['topsim_attributes_messages'] for i in range(len(evals))]), 3)
        topsim_af = round(np.mean([evals[i]['topsim_attributes_features'] for i in range(len(evals))]), 3)
        topsim_fm = round(np.mean([evals[i]['topsim_features_messages'] for i in range(len(evals))]), 3)
        
        if n_runs > 1: 
            topsim_am_std = round(np.std([evals[i]['topsim_attributes_messages'] for i in range(len(evals))]), 3)
            topsim_af_std = round(np.std([evals[i]['topsim_attributes_features'] for i in range(len(evals))]), 3)
            topsim_fm_std = round(np.std([evals[i]['topsim_features_messages'] for i in range(len(evals))]), 3)
        
        if '/' in name: 
            name = name.partition('/')[0]
        
        if n_runs == 1: 
            print(name, 'attribute-message', topsim_am, 
                  ', attribute-feature', topsim_af, 
                  ', feature-message', topsim_fm)
        elif n_runs > 1: 
            print(name, 'attribute-message', str(topsim_am) + '+-' + str(topsim_am_std),
                  ', attribute-feature', str(topsim_af) + '+-' + str(topsim_af_std),
                  ', feature-message', str(topsim_fm) + '+-' + str(topsim_fm_std)) 
    
        all_topsims_am.append(np.array([evals[i]['topsim_attributes_messages'] for i in range(len(evals))]))
    
    print(ttest_ind(all_topsims_am[0], all_topsims_am[-1]))


def show_rsas(names, vocab_size, message_length, path='./', mode='basic', n_vision=1, n_runs=1): 
    
    print('RSA')
    for name in names:
    
        _, _, evals, _ = get_stats(name, vocab_size, message_length, path, mode, n_vision, n_runs=n_runs)

        rsa_si = round(np.mean([evals[i]['rsa_sender_input'] for i in range(len(evals))]), 3)
        rsa_ri = round(np.mean([evals[i]['rsa_receiver_input'] for i in range(len(evals))]), 3)
        rsa_sr = round(np.mean([evals[i]['rsa_sender_receiver'] for i in range(len(evals))]), 3)
        
        if n_runs > 1: 
            rsa_si_std = round(np.std([evals[i]['rsa_sender_input'] for i in range(len(evals))]), 3)
            rsa_ri_std = round(np.std([evals[i]['rsa_receiver_input'] for i in range(len(evals))]), 3)
            rsa_sr_std = round(np.std([evals[i]['rsa_sender_receiver'] for i in range(len(evals))]), 3)
        
        if '/' in name: 
            name= name.partition('/')[0]
        
        if n_runs == 1: 
            print(name, 'sender-input', rsa_si, ', receiver-input', rsa_ri, ', sender-receiver', rsa_sr)
        elif n_runs > 1: 
            print(name,
                  ', nsender-input', str(rsa_si) + '+-' + str(rsa_si_std), 
                  ', receiver-input', str(rsa_ri) + '+-' + str(rsa_ri_std),
                  ', sender-receiver', str(rsa_sr) + '+-' + str(rsa_sr_std))


def show_groundedness_all(names, vocab_size, message_length, path='./', mode='basic', n_vision=1, n_runs=1): 
    
    print('groundedness')
    for name in names:
        print(name)
        
        _, _, evals, _ = get_stats(name, vocab_size, message_length, path, mode, n_vision, n_runs=n_runs)
        
        for run in range(n_runs): 
            print(run)
            print('posground', evals[run]['posground'])
            print('bosground', evals[run]['bosground'])


def show_groundedness(names, vocab_size, message_length, path='./', mode='basic', n_vision=1, n_runs=1): 
    
    print('groundedness')
    for name in names:
        
        _, _, evals, _ = get_stats(name, vocab_size, message_length, path, mode, n_vision, n_runs=n_runs)
        
        posground_return = {}
        bosground_return = {}
        for key in evals[0]['posground'].keys():
            if key != 'symbol_attribute' and key!= 'symbol_groundedness' and not 'pos' in key:
                if n_runs > 1: 
                    posground_return[key[0:-13]] = (str(round(np.mean([evals[i]['posground'][key] 
                                                                       for i in range(n_runs)]), 3)) 
                                                    + ' +-' +
                                                    str(round(np.std([evals[i]['posground'][key]
                                                                      for i in range(n_runs)]), 3)))
                else: 
                    posground_return[key[0:-13]] = round(evals[0]['posground'][key], 2)                           
        for key in evals[0]['bosground'].keys():
            if key != 'symbol_attribute' and key!= 'symbol_groundedness':
                if n_runs > 1: 
                    bosground_return[key[0:-13]] = (str(round(np.mean([evals[i]['bosground'][key] 
                                                                       for i in range(n_runs)]), 3)) 
                                                    + ' +-' +
                                                    str(round(np.std([evals[i]['bosground'][key]
                                                                      for i in range(n_runs)]), 3)))
                else: 
                    bosground_return[key[0:-13]] = round(evals[0]['bosground'][key], 2)
        
        if '/' in name: 
            name= name.partition('/')[0]
        print(name, '\nposground', posground_return, '\nbosground', bosground_return)
    

def get_eval_dict(names, vocab_size, message_length, path='./', mode='basic', n_vision=1): 
    
    for name in names: 
        path = path + mode + '/' + name + '/vs' + str(vocab_size) + '_ml' + str(message_length)

        eval_dicts = [] 

        for i in range(n_vision):
            if n_vision > 1: 
                append = str(i)
            else: 
                append = ''
            eval_dicts.append(pickle.load( open( path + '/eval' + append + '.pkl', 'rb' ) ))

        if n_vision == 1: 
            return eval_dicts[0]
        else: 
            return eval_dicts
            

def show_acquisition_speed(names, vocab_size, message_length, thresholds = [0.87, 0.9, 0.93], path='./',
                           mode='basic', n_runs=1):
    
    print("acquisition speed")
    
    for name_orig in names: 
        
        values_train = []
        values_val = []
        for run in range(n_runs): 

            name = name_orig + str(run)

            path_tmp = path + mode + '/' + name + '/vs' + str(vocab_size) + '_ml' + str(message_length)

            rewards = np.load(path_tmp + '/reward.npy')[1::2]
            try:
                val_rewards = np.load(path_tmp + '/val_reward.npy')[1::2]
            except:
                val_rewards = []

            epochs_train = np.zeros(len(thresholds))
            epochs_val = np.zeros(len(thresholds))

            for t, threshold in enumerate(thresholds): 
                try:
                    epoch = np.min(np.where(rewards >= threshold))
                    epochs_train[t] = epoch + 1
                except:
                    epochs_train[t] = np.nan
                try:
                    epoch = np.min(np.where(val_rewards >= threshold))
                    epochs_val[t] = epoch + 1
                except:
                    epochs_val[t] = np.nan
        
            values_train.append(epochs_train)
            values_val.append(epochs_val)
        
        mean_train = np.round(np.mean(values_train, axis=0), 3)
        std_train = np.round(np.std(values_train, axis=0), 3)
        mean_val = np.round(np.mean(values_val, axis=0), 3)
        std_val = np.round(np.std(values_val, axis=0), 3)
        
        if '/' in name:
                name = name.partition('/')[0]
        
        if n_runs == 1: 
            print(name_orig, "train", mean_train, ", val", mean_val)     
        else: 
            print(name_orig, "train", str(mean_train) + '+-' + str(std_train),
                  ", val", str(mean_val) + '+-' + str(std_val))


def ttests(name1, name2, vocab_size=4, message_length=3, path='./', mode='basic', n_vision=1, n_runs=10):
    
    R1, valR1, eval1, _ = get_stats(name1, vocab_size, message_length, path, mode, n_vision, n_runs=n_runs)
    R2, valR2, eval2, _ = get_stats(name2, vocab_size, message_length, path, mode, n_vision, n_runs=n_runs)
    _, _, eval1_zs, _ = get_stats(name1 + '_zs', vocab_size, message_length, path, mode, n_vision, n_runs=n_runs)
    _, _, eval2_zs, _ = get_stats(name2 + '_zs', vocab_size, message_length, path, mode, n_vision, n_runs=n_runs)
    
    # rewards
    ttest_R = ttest_ind([r[-1] for r in R1], [r[-1] for r in R2])
    ttest_valR = ttest_ind([r[-1] for r in valR1], [r[-1] for r in valR2])
    
    print("t test results, values rounded to 5 decimals", "\n")
    print("train reward", round(ttest_R[1], 5))
    print("validation reward", round(ttest_valR[1], 5))
    
    # acquisition speed 
    speed = []
    for R in [R1, R2]:
        epochs_train = np.zeros((3, n_runs))
        for run in range(n_runs):
            for t, threshold in enumerate([0.87, 0.9, 0.93]): 
                epoch = np.min(np.where(R[run] >= threshold))
                epochs_train[t, run] = epoch + 1
        speed.append(epochs_train)
    
    ttest_speed = [
        ttest_ind(speed[0][0, :], speed[1][0, :]),
        ttest_ind(speed[0][1,:], speed[1][1, :]),
        ttest_ind(speed[0][2, :], speed[1][2, :])
    ]
    
    print("speed 0.87", round(ttest_speed[0][1], 5))
    print("speed 0.90", round(ttest_speed[1][1], 5))
    print("speed 0.93", round(ttest_speed[2][1], 5))    

    # topsim
    topsim1 = [eval1[i]['topsim_attributes_messages'] for i in range(n_runs)]
    topsim2 = [eval2[i]['topsim_attributes_messages'] for i in range(n_runs)]
    print("topsim", round(ttest_ind(topsim1, topsim2)[1], 5))
    topsim1 += [eval1_zs[i]['topsim_attributes_messages'] for i in range(n_runs)]
    topsim2 += [eval2_zs[i]['topsim_attributes_messages'] for i in range(n_runs)]
    print("topsim extended by values from zero-shot runs", round(ttest_ind(topsim1, topsim2)[1], 5))
    
    # zero-shot
    zs1 = [eval1_zs[i]['zero_shot_reward'] for i in range(n_runs)]
    zs2 = [eval2_zs[i]['zero_shot_reward'] for i in range(n_runs)]
    print("zero-shot", round(ttest_ind(zs1, zs2)[1], 5))
    
    # RSA
    rsa_si1 = [eval1[i]['rsa_sender_input'] for i in range(n_runs)]
    rsa_ri1 = [eval1[i]['rsa_receiver_input'] for i in range(n_runs)]
    rsa_sr1 = [eval1[i]['rsa_sender_receiver'] for i in range(n_runs)]
    rsa_si2 = [eval2[i]['rsa_sender_input'] for i in range(n_runs)]
    rsa_ri2 = [eval2[i]['rsa_receiver_input'] for i in range(n_runs)]
    rsa_sr2 = [eval2[i]['rsa_sender_receiver'] for i in range(n_runs)]

    print("rsa sender-input", round(ttest_ind(rsa_si1, rsa_si2)[1], 5))
    print("rsa receiver-input", round(ttest_ind(rsa_ri1, rsa_ri2)[1], 5))
    print("rsa sender-receiver", round(ttest_ind(rsa_sr1, rsa_sr2)[1], 5))
