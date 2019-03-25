from PatientVec.Experiments.base_config_gen import *

def LSTM(data, structured, args) :
    encoder_params = rnn_encoder_params(rnntype='lstm', hidden_size=128, args=args)
    config = seq_classifier_experiment(data, encoder_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config    

def Average(data, structured, args) :
    encoder_params = average_encoder_params(projection=True, hidden_size=256, activation='relu', args=args)
    config = seq_classifier_experiment(data, encoder_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config    

def CNN(data, structured, args) :
    encoder_params = cnn_encoder_params(hidden_size=64, kernel_sizes=[3, 5, 7, 9], activation='relu', args=args)
    config = seq_classifier_experiment(data, encoder_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def SRU(data, structured, args) :
    encoder_params = rnn_encoder_params(rnntype='sru', hidden_size=128, args=args)
    config = seq_classifier_experiment(data, encoder_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config 

def LSTM_with_attention(data, structured, args) :
    encoder_params = rnn_encoder_params(rnntype='lstm', hidden_size=128, args=args)
    attention_params = add_attention(sim_type='additive', hidden_size=128, args=args)
    config = seq_classifier_with_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def Average_with_attention(data, structured, args) :
    encoder_params = average_encoder_params(projection=True, hidden_size=256, activation='relu', args=args)
    attention_params = add_attention(sim_type='additive', hidden_size=128, args=args)
    config = seq_classifier_with_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def CNN_with_attention(data, structured, args) :
    encoder_params = cnn_encoder_params(hidden_size=64, kernel_sizes=[3, 5, 7, 9], activation='relu', args=args)
    attention_params = add_attention(sim_type='additive', hidden_size=128, args=args)
    config = seq_classifier_with_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def SRU_with_attention(data, structured, args) :
    encoder_params = rnn_encoder_params(rnntype='sru', hidden_size=128, args=args)
    attention_params = add_attention(sim_type='additive', hidden_size=128, args=args)
    config = seq_classifier_with_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

vanilla_configs = [Average, LSTM, CNN]
attention_configs = [Average_with_attention, LSTM_with_attention, CNN_with_attention]

def Hierarchical_LSTM_with_attention(data, structured, args) :
    word_encoder_params = rnn_encoder_params(rnntype='lstm', hidden_size=128, args=args)
    word_attention_params = add_attention(sim_type='additive', hidden_size=128, args=args)

    sentence_encoder_params = rnn_encoder_params(rnntype='lstm', hidden_size=128, args=args)
    sentence_attention_params = add_attention(sim_type='additive', hidden_size=128, args=args)

    config = hierarchical_experiment(data, word_encoder_params, sentence_encoder_params, word_attention_params, sentence_attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)

    config['training_config']['common']['bsize'] = 16
    return config

hierarchical_configs =  [Hierarchical_LSTM_with_attention]

def LSTM_with_conditional_attention(data, structured, encodings, args) :
    encoder_params = rnn_encoder_params(rnntype='lstm', hidden_size=128, args=args)
    attention_params = add_structured_attention(encodings, data.get_encodings_dim(encodings), sim_type='additive', hidden_size=128, args=args)
    config = seq_classifier_with_structured_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def CNN_with_conditional_attention(data, structured, encodings, args) :
    encoder_params = cnn_encoder_params(hidden_size=64, kernel_sizes=[3, 5, 7, 9], activation='relu', args=args)
    attention_params = add_structured_attention(encodings, data.get_encodings_dim(encodings), sim_type='additive', hidden_size=128, args=args)
    config = seq_classifier_with_structured_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def Average_with_conditional_attention(data, structured, encodings, args) :
    encoder_params = average_encoder_params(projection=True, hidden_size=256, activation='relu', args=args)
    attention_params = add_structured_attention(encodings, data.get_encodings_dim(encodings), sim_type='additive', hidden_size=128, args=args)
    config = seq_classifier_with_structured_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def SRU_with_conditional_attention(data, structured, encodings, args) :
    encoder_params = rnn_encoder_params(rnntype='sru', hidden_size=128, args=args)
    attention_params = add_structured_attention(encodings, data.get_encodings_dim(encodings), sim_type='additive', hidden_size=128, args=args)
    config = seq_classifier_with_structured_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

structured_configs = [Average_with_conditional_attention, LSTM_with_conditional_attention, CNN_with_conditional_attention]

def LR(data, structured, args) :
    decoder_params = add_decoder([], [], args)
    config = vector_experiment(data, decoder_params)
    config['model']['reg'] = { "type" : "l1", "weight" : 1e-5 }
    for g in config['training_config']['groups'] :
        g[1]['weight_decay'] = 0.0
        
    config['training_config']['common']['bsize'] = 128
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def MLP(data, structured, args) :
    decoder_params = add_decoder([256], ['tanh'], args)
    config = vector_experiment(data, decoder_params)
    config['training_config']['common']['bsize'] = 128
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

vector_configs = [LR, MLP]