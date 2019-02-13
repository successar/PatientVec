from PatientVec.configs import *

def LSTM(data, structured) :
    encoder_params = lstm_encoder_params(hidden_size=128)
    config = seq_classifier_experiment(data, encoder_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config    

def Average(data, structured) :
    encoder_params = average_encoder_params(projection=True, hidden_size=256, activation='relu')
    config = seq_classifier_experiment(data, encoder_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config    

def CNN(data, structured) :
    encoder_params = cnn_encoder_params(hidden_size=64, kernel_sizes=[3, 5, 7, 9], activation='relu')
    config = seq_classifier_experiment(data, encoder_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def LSTM_with_attention(data, structured) :
    encoder_params = lstm_encoder_params(hidden_size=128)
    attention_params = add_attention(sim_type='additive', hidden_size=128)
    config = seq_classifier_with_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def Average_with_attention(data, structured) :
    encoder_params = average_encoder_params(projection=True, hidden_size=256, activation='relu')
    attention_params = add_attention(sim_type='additive', hidden_size=128)
    config = seq_classifier_with_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def CNN_with_attention(data, structured) :
    encoder_params = cnn_encoder_params(hidden_size=64, kernel_sizes=[3, 5, 7, 9], activation='relu')
    attention_params = add_attention(sim_type='additive', hidden_size=128)
    config = seq_classifier_with_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

basic_configs = [Average, LSTM, CNN, Average_with_attention, LSTM_with_attention, CNN_with_attention]

def Hierarchical_LSTM_with_attention(data, structured) :
    word_encoder_params = lstm_encoder_params(hidden_size=128)
    word_attention_params = add_attention(sim_type='additive', hidden_size=128)

    sentence_encoder_params = lstm_encoder_params(hidden_size=128)
    sentence_attention_params = add_attention(sim_type='additive', hidden_size=128)

    config = hierarchical_experiment(data, word_encoder_params, sentence_encoder_params, word_attention_params, sentence_attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)

    config['training_config']['common']['bsize'] = 16
    return config

hierarchical_configs =  [Hierarchical_LSTM_with_attention]

def LSTM_with_conditional_attention(data, structured, encodings) :
    encoder_params = lstm_encoder_params(hidden_size=128)
    attention_params = add_structured_attention(encodings, data.get_encodings_dim(encodings), sim_type='additive', hidden_size=128)
    config = seq_classifier_with_structured_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def CNN_with_conditional_attention(data, structured, encodings) :
    encoder_params = cnn_encoder_params(hidden_size=64, kernel_sizes=[3, 5, 7, 9], activation='relu')
    attention_params = add_structured_attention(encodings, data.get_encodings_dim(encodings), sim_type='additive', hidden_size=128)
    config = seq_classifier_with_structured_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

def Average_with_conditional_attention(data, structured, encodings) :
    encoder_params = average_encoder_params(projection=True, hidden_size=256, activation='relu')
    attention_params = add_structured_attention(encodings, data.get_encodings_dim(encodings), sim_type='additive', hidden_size=128)
    config = seq_classifier_with_structured_attention_experiment(data, encoder_params, attention_params)
    if structured :
        config = make_structured(config, data.structured_dim)
    return config

structured_configs = [Average_with_conditional_attention, LSTM_with_conditional_attention, CNN_with_conditional_attention]