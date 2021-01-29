




def generalized_cosine(A,B,speedup=True):
    cos_AB = np.sum(inner1d(A, B)) / \
        np.sqrt( np.sum(inner1d(A, A)) * np.sum(inner1d(B, B)) )
    return cos_AB

def calc_R(x,normalize=True,white_noise=0.01):
    if normalize == True:
        x = x / np.sqrt(np.sum(x**2,axis=1)[:,np.newaxis])
    result = np.tensordot(x,x,axes=[0,0]) / np.shape(x)[0]
    if white_noise > 0:
        unit = np.diag(np.ones(np.shape(x)[1]))
        result = (1-white_noise)*result + white_noise/np.shape(x)[1]*unit
    return result

def sort_data(data, labels):
    data_sorted = dict()
    for i in range(0,10):
        data_sorted[str(i)] = []
    for (idx,entry) in enumerate(data):
        data_sorted[str(labels[idx])].append(list(entry))
    
    for k in data_sorted.keys():
        data_sorted[k] = np.array(data_sorted[k])
    
    return data_sorted

def gather_activations(base_model, data_in):
    activations = dict()
    for k in data_in.keys():
        activations[k] = []
        model = not_ShallowNet.build(width=28, height=28, depth=1, classes=10)
        for layer in base_model.layers:
            layer_info = layer.get_config()
            if layer_info['name'].find('dense') >= 0 or \
                    layer_info['name'].find('flatten') >= 0:
                model = Model(base_model.input, outputs=layer.output)
                out = model.predict(data_in[k])
                activations[k].append(out)

    return activations
