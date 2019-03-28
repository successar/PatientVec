from PatientVec.models.Baseline import ClassificationTrainer as VectorCT
from PatientVec.models.Vanilla import ClassificationTrainer as BasicCT
from PatientVec.models.Hierarchical import ClassificationTrainer as HierCT
from PatientVec.trainer import Trainer, Evaluator
from PatientVec.Experiments.modifiable_config_exp import *
from PatientVec.common import *

def get_basic_data(data, truncate=90, structured=True, encodings=None) :
    train_data = data.filter_data_length(data.get_data('train', structured=structured, encodings=encodings), truncate=truncate)
    dev_data = data.filter_data_length(data.get_data('dev', structured=structured, encodings=encodings), truncate=truncate)
    test_data = data.filter_data_length(data.get_data('test', structured=structured, encodings=encodings), truncate=truncate)
    return train_data, dev_data, test_data

def basic_experiments(data, configs, args) :
    do_test = vars(args).get('do_test', False)
    do_all = vars(args).get('do_all', False)
    structured = vars(args).get('structured', True)
    train_data, dev_data, test_data = get_basic_data(data, structured=structured, truncate=90)

    for e in configs :
        config = e(data, structured=structured, args=args)
        if args.output_dir is not None :
            config['exp_config']['basepath'] = args.output_dir
        if hasattr(args, 'modify_config') :
            config = args.modify_config(config)
        # config['training_config']['common']['bsize'] = 16
        print(config)

        trainer = Trainer(BasicCT, config, _type=data.metrics_type, display_metrics=args.display)
            
        if not do_test :
            n_iters = vars(args).get('n_iters', 10)
            trainer.train(train_data, dev_data, n_iters=n_iters, save_on_metric=data.save_on_metric)
            dirname = [trainer.model.dirname]
        else :
            if do_all :
                dirname = get_all_model(os.path.dirname(trainer.model.dirname))
            else :
                dirname = [get_latest_model(os.path.dirname(trainer.model.dirname))]

        print(len(dirname))
        for d in dirname[:10] :
            if do_test and has_test_results(d): continue

            evaluator = Evaluator(BasicCT, d, _type=data.metrics_type, display_metrics=args.display)
            evaluator.evaluate(dev_data, test_data, save_results=True)
        print('='*300)
        
def structured_attention_experiments(data, configs, args) :
    do_test = vars(args).get('do_test', False)
    structured = vars(args).get('structured', True)
    train_data, dev_data, test_data = get_basic_data(data, structured=structured, truncate=90, encodings=data.structured_columns)

    for e in configs :
        config = e(data, structured=structured, encodings=data.structured_columns, args=args)
        if args.output_dir is not None :
            config['exp_config']['basepath'] = args.output_dir
        if hasattr(args, 'modify_config') :
            config = args.modify_config(config)
        print(config)

        trainer = Trainer(BasicCT, config, _type=data.metrics_type, display_metrics=args.display)
        
        if not do_test :
            n_iters = vars(args).get('n_iters', 10)
            trainer.train(train_data, dev_data, n_iters=n_iters, save_on_metric=data.save_on_metric)
            dirname = trainer.model.dirname
        else :
            dirname = get_latest_model(os.path.dirname(trainer.model.dirname))

        if do_test and has_test_results(dirname): continue

        evaluator = Evaluator(BasicCT, dirname, _type=data.metrics_type, display_metrics=args.display)
        evaluator.evaluate(dev_data, test_data, save_results=True)
        print('='*300)

def training_size_experiments(data, configs, args) :
    do_test = vars(args).get('do_test', False)
    structured = vars(args).get('structured', True)
    train_data, dev_data, test_data = get_basic_data(data, structured=structured, truncate=90)
    np.random.seed(args.seed)
    if len(train_data.X) < args.n :
        print("Less train than n", args.n, len(train_data.X))
        return
    train_data = train_data.sample(n=args.n)

    for e in configs :
        config = e(data, structured=structured, args=args)
        if args.output_dir is not None :
            config['exp_config']['basepath'] = args.output_dir + '/ts_experiments/n=' + str(args.n)
        if hasattr(args, 'modify_config') :
            config = args.modify_config(config)
        print(config)

        trainer = Trainer(BasicCT, config, _type=data.metrics_type, display_metrics=args.display)

        if not do_test :
            n_iters = vars(args).get('n_iters', 10)
            trainer.train(train_data, dev_data, n_iters=n_iters, save_on_metric=data.save_on_metric)
            dirname = trainer.model.dirname
        else :
            dirname = get_latest_model(os.path.dirname(trainer.model.dirname))

        if do_test and has_test_results(dirname): continue

        evaluator = Evaluator(BasicCT, dirname, _type=data.metrics_type, display_metrics=args.display)
        evaluator.evaluate(dev_data, test_data, save_results=True)
        print('='*300)

def diagnosis_pretrained_experiments(data, args) :
    do_test = vars(args).get('do_test', False)
    structured = True
    train_data, dev_data, test_data = get_basic_data(data, structured=structured, truncate=90)
    if 'n' in vars(args) :
        print("Sampling ", args.n)
        np.random.seed(args.seed)
        train_data = train_data.sample(n=args.n)

    config = LSTM(data, structured=structured, args=args)
    if args.output_dir is not None :
        config['exp_config']['basepath'] = args.output_dir
        if 'n' in vars(args) :
            config['exp_config']['basepath'] = args.output_dir + '/ts_experiments/n=' + str(args.n)
    if hasattr(args, 'modify_config') :
        config = args.modify_config(config)

    config['exp_config']['exp_name'] += '+Pretrained'
    print(config)

    trainer = Trainer(BasicCT, config, _type=data.metrics_type, display_metrics=args.display)
    phenotype_model = get_latest_model(os.path.join('outputs', data.name + '_hcup', 'Basic', 'LSTM(hs=128)+Structured'))
    if phenotype_model is None :
        raise LookupError("Phenotype model not available")
    trainer.load_pretrained_model(phenotype_model)

    if not do_test :
        n_iters = vars(args).get('n_iters', 10)
        trainer.train(train_data, dev_data, n_iters=n_iters, save_on_metric=data.save_on_metric)
        dirname = trainer.model.dirname
    else :
        dirname = get_latest_model(os.path.dirname(trainer.model.dirname))

    if do_test and has_test_results(dirname): return

    evaluator = Evaluator(BasicCT, dirname, _type=data.metrics_type, display_metrics=args.display)
    evaluator.evaluate(dev_data, test_data, save_results=True)
    print('='*300)
            
def hierarchical_experiments(data, args) :
    do_test = vars(args).get('do_test', False)
    structured = vars(args).get('structured', True)
    train_data, dev_data, test_data = get_basic_data(data, structured=structured, truncate=90)
    
    for e in hierarchical_configs :
        config = e(data, structured=structured, args=args)
        if args.output_dir is not None :
            config['exp_config']['basepath'] = args.output_dir
        if hasattr(args, 'modify_config') :
            config = args.modify_config(config)
            
        config['training_config']['common']['bsize'] = vars(args).get('bsize', 16)
        print(config)
        
        trainer = Trainer(HierCT, config, _type=data.metrics_type, display_metrics=args.display)

        if not do_test :
            n_iters = vars(args).get('n_iters', 10)
            trainer.train(train_data, dev_data, n_iters=n_iters, save_on_metric=data.save_on_metric)
            dirname = trainer.model.dirname
        else :
            dirname = get_latest_model(os.path.dirname(trainer.model.dirname))

        if do_test and has_test_results(dirname): continue

        evaluator = Evaluator(HierCT, dirname, _type=data.metrics_type, display_metrics=args.display)
        evaluator.evaluate(dev_data, test_data, save_results=True)
        print('='*300)
        
from PatientVec.models.baselines.LR import LR

def lr_experiments(data, args) :
    structured = vars(args).get('structured', True)
    structured = structured & (data.structured_dim > 0)
    train_data, dev_data, test_data = get_basic_data(data, structured=structured, truncate=90)
    for norm in [None, 'l1', 'l2'] :
        config = {'vocab' : data.vocab, 
                  'stop_words' : True, 
                  'exp_name' : data.name, 
                  'type' : data.metrics_type, 
                  'norm' : norm, 
                  'constant_mul' : 1.0, 
                  'has_structured' : structured, 
                  'lda' : False, 
                  'methods' : ['count', 'binary', 'tfidf'],
                  'only_structured' : False,
                  'basepath' : args.output_dir}
        lr = LR(config)
        print(config)
        print(lr.has_structured)
        lr.train(train_data)
        lr.evaluate(name='dev', data=dev_data, save_results=True)
        lr.evaluate(name='test', data=test_data, save_results=True)
        
def lda_experiments(data, args) :
    structured = vars(args).get('structured', True)
    train_data, dev_data, test_data = get_basic_data(data, structured=structured, truncate=90)
    config = {'vocab' : data.vocab, 
                  'stop_words' : True, 
                  'exp_name' : data.name, 
                  'type' : data.metrics_type, 
                  'norm' : None, 
                  'constant_mul' : 1.0, 
                  'has_structured' : structured, 
                  'lda' : True, 
                  'methods' : ['count', 'binary', 'tfidf'],
                  'only_structured' : False,
                  'basepath' : args.output_dir}
    lr = LR(config)
    print(config)
    print(lr.has_structured)
    lr.train_lda(train_data)
    lr.evaluate_lda(name='dev', data=dev_data, save_results=True)
    lr.evaluate_lda(name='test', data=test_data, save_results=True)

experiment_types = {
    'vanilla' : lambda data, args : basic_experiments(data, vanilla_configs, args),
    'attention' : lambda data, args : basic_experiments(data, attention_configs, args),
    'basic_sru' : lambda data, args : basic_experiments(data, [SRU, SRU_with_attention], args),
    'hierarchical' : hierarchical_experiments,
    'structured' : lambda data, args : structured_attention_experiments(data, structured_configs, args),
    'structured_sru' : lambda data, args : structured_attention_experiments(data, [SRU_with_conditional_attention], args),
    'lr' : lr_experiments,
    'basic' : basic_experiments,
    'ts_experiments' : lambda d, a : training_size_experiments(d, [Average, LSTM, LSTM_with_attention], a),
    'pretrained' : diagnosis_pretrained_experiments,
    'lda' : lda_experiments
}

