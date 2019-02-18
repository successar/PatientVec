def add_embedder(vocab_size, embed_size, embedding_file) :
    return {
        'type' : "token_embedder" ,
        "vocab_size" : vocab_size,
        "embed_size" : embed_size,
        "embedding_file" : embedding_file
    }

##################### Encoder Units #######################################################

def lstm_encoder_params(hidden_size=128) :
    return {
        'exp_name' : 'LSTM' + '(hs=' + str(hidden_size) + ')',
        'params' : {
            "type" : 'lstm',
            "hidden_size" : hidden_size
        }
    }

def cnn_encoder_params(hidden_size=64, kernel_sizes=[3, 5, 7, 9], activation='relu') :
    return {
        'exp_name' : 'CNN' + '(hs=' + str(hidden_size) + ')(kernels=' + ",".join(map(str, kernel_sizes)) + ')',
        'params' :  { 'type' : 'cnn', 'hidden_size' : hidden_size, 'kernel_sizes' : kernel_sizes, 'activation' : activation}
    }
       

def average_encoder_params(projection=True, hidden_size=256, activation='relu') :
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

def add_attention(sim_type='additive', hidden_size=128) :
    return  {    
        'exp_name' : 'Attention(' + sim_type + ')(hs=' + str(hidden_size) + ')',
        'params' : {
                "similarity" : {
                "type" : sim_type,
                "hidden_size" : hidden_size
            }
        }
    }

def add_structured_attention(encodings, nconditional, sim_type='additive', hidden_size=128) :
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