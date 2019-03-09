from PatientVec.Experiments.config_units import *

def get_basic_model_config(data, exp_name) :
    config = {
        "model" : {
            "type" : None,
            "embedder" : add_embedder(**data.get_embedding_params()), 
            "decoder" : {
                "num_layers" : 2, 
                "hidden_dims" : [128, data.output_size], 
                "activations" : ["tanh", "linear"]
            },
            "predictor" : {
                "type" : data.predictor_type
            },
            "structured" : {
                "use_structured" : False
            }
        },
        "training_config" : {
            "type" : "Adam",
            "groups" : [
                ( r".*", {"lr" : 0.001, "weight_decay" : 1e-5})
            ]
            ,
            "common" : {
                "bsize" : 32,
                "class_weight" : True
            }
        },
        "exp_config" : {
            "exp_name" : exp_name
        }
    }

    return config

def make_structured(config, structured_dim) :
    config['exp_config']['exp_name'] += '+Structured'
    config['model']['structured'] = {
        "use_structured" : True,
        "structured_dim" : structured_dim
    }

    return config

def modify_training_params(config, optimiser, lr, weight_decay) :
    config['training_config']['type'] = optimiser
    for g in config['training_config']['groups'] :
        g[1]['lr'] = lr
        g[1]['weight_decay'] = weight_decay

    config['exp_config']['exp_name'] += '/hyperparams/lr=' + str(lr) + '+weight_decay=' + str(weight_decay)

    return config

def replicate_config(config, alpha) :
    config['model']['predictor']['replicate'] = True
    config['model']['predictor']['alpha'] = alpha
    config['exp_config']['exp_name'] += '+replicate(alpha=+'+str(alpha)+')'
    return config

############### Seq Classifier Experiment ###################################

def seq_classifier_experiment(data, encoder_params) :
    config = get_basic_model_config(data, data.name + '/Basic/' + encoder_params['exp_name'])
    config['model']['type'] = "seq_classifier"
    config['model']['encoder'] = encoder_params['params']
    return config

################## Seq Classifier with Attention ###################################

def seq_classifier_with_attention_experiment(data, encoder_params, attention_params) :
    config = get_basic_model_config(data, data.name + '/Attention/' + '+'.join([encoder_params['exp_name'], attention_params['exp_name']]))
    config['model']['type'] = "seq_classifier_with_attention"
    config['model']['encoder'] = encoder_params['params']
    config['model']['attention'] = attention_params['params']
    return config

######################### Seq Classifier with Structured Attention #######################

def seq_classifier_with_structured_attention_experiment(data, encoder_params, attention_params) :
    config = get_basic_model_config(data, data.name + '/Structured Attention/' + '+'.join([encoder_params['exp_name'], attention_params['exp_name']]))
    config['model']['type'] = "seq_classifier_with_structured_attention"
    config['model']['encoder'] = encoder_params['params']
    config['model']['attention'] = attention_params['params']
    return config

########################### Hierarchical Model #################################################

def hierarchical_experiment(data, word_encoder_params, sentence_encoder_params, word_attention_params, sentence_attention_params) : 
    config = get_basic_model_config(data, data.name + '/Hierarchical Attention/') 
    config['exp_config']['exp_name'] += '+'.join([word_encoder_params['exp_name'], word_attention_params['exp_name'], 
                                                sentence_encoder_params['exp_name'], sentence_attention_params['exp_name']])

    config['model']['type'] = "hierarchical_classifier_with_attention"
    config['model']['word_encoder'] = word_encoder_params['params']
    config['model']['word_attention'] = word_attention_params['params']
    config['model']['sentence_encoder'] = sentence_encoder_params['params']
    config['model']['sentence_attention'] = sentence_attention_params['params']
    return config

def vector_experiment(data) :
    config = get_basic_model_config(data, data.name + '/Test_TFIDF')
    config['model']['type'] = 'vec_classifier'
    del config['model']['embedder']
    config['model']['decoder']['input_dim'] = len(data.bowder.words_to_keep)
    config['model']['decoder']['num_layers'] = 1
    config['model']['decoder']['hidden_dims'] = [data.output_size]
    config['model']['decoder']['activations'] = ['linear']
    config['training_config']['common']['bsize'] = 256
    config['model']['reg'] = { "type" : "l1", "weight" : 0.5}
    config = make_structured(config, data.structured_dim)
    config = modify_training_params(config, 'Adam', 0.001, 0.0)
    return config



