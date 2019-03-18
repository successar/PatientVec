def add_embedder(vocab_size, embed_size, embedding_file) :
    return {
        'type' : "token_embedder" ,
        "vocab_size" : vocab_size,
        "embed_size" : embed_size,
        "embedding_file" : embedding_file
    }

def add_elmo_embedder(vocab_size, embed_size, elmo_options) :
    return {
        'type' : 'elmo_embedder',
        'vocab_size' : vocab_size,
        'embed_size' : embed_size,
        'elmo_options' : elmo_options
    }

##################### Encoder Units #######################################################

def rnn_encoder_params(rnntype='lstm', hidden_size=128, args=None) :
    if args is not None and 'encoder' in vars(args) :
        rnntype = vars(args)['encoder'].get('rnntype', rnntype)
        hidden_size = vars(args)['encoder'].get('hidden_size', hidden_size)

    return {
        'exp_name' : rnntype.upper() + '(hs=' + str(hidden_size) + ')',
        'params' : {
            "type" : rnntype,
            "hidden_size" : hidden_size
        }
    }

def cnn_encoder_params(hidden_size=64, kernel_sizes=[3, 5, 7, 9], activation='relu', args=None) :
    if args is not None and 'encoder' in vars(args) :
        hidden_size = vars(args)['encoder'].get('hidden_size', hidden_size)
        kernel_sizes = vars(args)['encoder'].get('kernel_sizes', kernel_sizes)
        activation = vars(args)['encoder'].get('activation', activation)

    return {
        'exp_name' : 'CNN' + '(hs=' + str(hidden_size) + ')(kernels=' + ",".join(map(str, kernel_sizes)) + ')',
        'params' :  { 'type' : 'cnn', 'hidden_size' : hidden_size, 'kernel_sizes' : kernel_sizes, 'activation' : activation}
    }
       

def average_encoder_params(projection=True, hidden_size=256, activation='relu', args=None) :
    if args is not None and 'encoder' in vars(args) :
        hidden_size = vars(args)['encoder'].get('hidden_size', hidden_size)
        activation = vars(args)['encoder'].get('activation', activation)

    return {
        'exp_name' : 'Average' + '(hs=' + str(hidden_size) + ')',
        'params' : {
            'type' : 'average',
            'projection' : projection,
            'hidden_size' : hidden_size,
            'activation' : activation
        }
    }

#################### Attention Units #########################################################

def add_attention(sim_type='additive', hidden_size=128, args=None) :
    if args is not None and 'attention' in vars(args) :
        sim_type = vars(args)['attention'].get('sim_type', sim_type)
        hidden_size = vars(args)['attention'].get('hidden_size', hidden_size)

    return  {    
        'exp_name' : 'Attention(' + sim_type + ')(hs=' + str(hidden_size) + ')',
        'params' : {
                "similarity" : {
                "type" : sim_type,
                "hidden_size" : hidden_size
            }
        }
    }

def add_structured_attention(encodings, nconditional, sim_type='additive', hidden_size=128, args=None) :
    if args is not None and 'attention' in vars(args) :
        sim_type = vars(args)['attention'].get('sim_type', sim_type)
        hidden_size = vars(args)['attention'].get('hidden_size', hidden_size)

    return {    
        'exp_name' : 'Attention(' + sim_type + ')(' + ('.'.join(encodings) if len(encodings) < 10 else 'all') + ')' + '(hs=' + str(hidden_size) + ')',
        'params' : {
                "similarity" : {
                "type" : sim_type,
                "hidden_size" : hidden_size,
                'tensor_2_dim' : nconditional
            }
        }
    }

def add_decoder(hidden_dims, activations, args=None) :
    if args is not None and 'decoder' in vars(args) :
        hidden_dims = vars(args)['decoder'].get('hidden_dims', hidden_dims)
        activations = vars(args)['decoder'].get('activations', activations)
        
    return {
        'exp_name' : 'Decoder(' + (','.join([str(x)+'.'+str(y) for x, y in zip(hidden_dims, activations)])) + ')',
        'params' : {
            'num_layers' : len(hidden_dims) + 1,
            'hidden_dims' : hidden_dims,
            'activations' : activations + ['linear'],
            'dropout' : 0.2
        }
    }