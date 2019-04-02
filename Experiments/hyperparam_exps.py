from PatientVec.models.Baseline import ClassificationTrainer as VectorCT
from PatientVec.models.Vanilla import ClassificationTrainer as BasicCT
from PatientVec.models.Hierarchical import ClassificationTrainer as HierCT
from PatientVec.trainer import Trainer, Evaluator
from PatientVec.Experiments.modifiable_config_exp import *

def get_basic_data(data, truncate=90, structured=True, encodings=None) :
    train_data = data.filter_data_length(data.get_data('train', structured=structured, encodings=encodings), truncate=truncate)
    dev_data = data.filter_data_length(data.get_data('dev', structured=structured, encodings=encodings), truncate=truncate)

    return train_data, dev_data
    
def vanilla_experiments(data, args) :
    structured = vars(args).get('structured', True)
    train_data, dev_data = get_basic_data(data, structured=structured, truncate=90)

    for e in vanilla_configs :
        config = e(data, structured=structured, args=args)
        if args.output_dir is not None :
            config['exp_config']['basepath'] = args.output_dir
        if hasattr(args, 'modify_config') :
            config = args.modify_config(config)
        print(config)

        trainer = Trainer(BasicCT, config, _type=data.metrics_type, display_metrics=args.display)
        trainer.train(train_data, dev_data, save_on_metric=data.save_on_metric)

        evaluator = Evaluator(BasicCT, trainer.model.dirname, _type=data.metrics_type, display_metrics=args.display)
        _ = evaluator.evaluate(dev_data, save_results=True)
        print('='*300)

def attention_experiments(data, args) :
    structured = vars(args).get('structured', True)
    train_data, dev_data = get_basic_data(data, structured=structured, truncate=90)

    for e in attention_configs :
        config = e(data, structured=structured, args=args)
        if args.output_dir is not None :
            config['exp_config']['basepath'] = args.output_dir
        if hasattr(args, 'modify_config') :
            config = args.modify_config(config)
        print(config)

        trainer = Trainer(BasicCT, config, _type=data.metrics_type, display_metrics=args.display)
        trainer.train(train_data, dev_data, save_on_metric=data.save_on_metric)

        evaluator = Evaluator(BasicCT, trainer.model.dirname, _type=data.metrics_type, display_metrics=args.display)
        _ = evaluator.evaluate(dev_data, save_results=True)
        print('='*300)
            
def hierarchical_experiments(data, args) :
    structured = vars(args).get('structured', True)
    train_data, dev_data = get_basic_data(data, structured=structured, truncate=90)

    for e in hierarchical_configs :
        config = e(data, structured=structured, args=args)
        if args.output_dir is not None :
            config['exp_config']['basepath'] = args.output_dir
        if hasattr(args, 'modify_config') :
            config = args.modify_config(config)
        print(config)
        
        trainer = Trainer(HierCT, config, _type=data.metrics_type, display_metrics=args.display)
        trainer.train(train_data, dev_data, n_iters=10, save_on_metric=data.save_on_metric)

        evaluator = Evaluator(HierCT, trainer.model.dirname, _type=data.metrics_type, display_metrics=args.display)
        _ = evaluator.evaluate(dev_data, save_results=True)
        print('='*300)

def structured_attention_experiments(data, args) :
    structured = vars(args).get('structured', True)
    train_data, dev_data = get_basic_data(data, structured=structured, truncate=90, encodings=data.structured_columns)

    for e in structured_configs :
        config = e(data, structured=structured, encodings=data.structured_columns, args=args)
        if args.output_dir is not None :
            config['exp_config']['basepath'] = args.output_dir
        if hasattr(args, 'modify_config') :
            config = args.modify_config(config)
        print(config)

        trainer = Trainer(BasicCT, config, _type=data.metrics_type, display_metrics=args.display)
        trainer.train(train_data, dev_data, save_on_metric=data.save_on_metric)

        evaluator = Evaluator(BasicCT, trainer.model.dirname, _type=data.metrics_type, display_metrics=args.display)
        _ = evaluator.evaluate(dev_data, save_results=True)
        print('='*300)
        
def vector_experiments(data, args) :
    structured = vars(args).get('structured', True)
    train_data, dev_data = get_basic_data(data, structured=structured, truncate=100)
    data.generate_bowder(train_data, stop_words=True, norm=args.norm)
    train_data.X = data.get_vec_encoding(train_data, _type=args.bow_type)
    dev_data.X = data.get_vec_encoding(dev_data, _type=args.bow_type)
    
    for e in vector_configs :
        config = e(data, structured=structured, args=args)
        if args.output_dir is not None :
            config['exp_config']['basepath'] = args.output_dir
        if hasattr(args, 'modify_config') :
            config = args.modify_config(config)
        print(config)

        trainer = Trainer(VectorCT, config, _type=data.metrics_type, display_metrics=args.display)
        trainer.train(train_data, dev_data, save_on_metric=data.save_on_metric)

        evaluator = Evaluator(VectorCT, trainer.model.dirname, _type=data.metrics_type, display_metrics=args.display)
        _ = evaluator.evaluate(dev_data, save_results=True)
        print('='*300)
        
from PatientVec.models.baselines.LR import LR

def lr_experiments(data, args) :
    train_data, dev_data = get_basic_data(data, truncate=90)
    config = {'vocab' : data.vocab, 'stop_words' : True, 'exp_name' : data.name, 'type' : data.metrics_type}
    config.update(vars(args).get('lr', {}))
    lr = LR(config)
    lr.train(train_data)
    lr.evaluate(dev_data, save_results=True)

experiment_types = {
    'vanilla' : vanilla_experiments,
    'attention' : attention_experiments,
    'hierarchical' : hierarchical_experiments,
    'structured' : structured_attention_experiments,
    'lr' : lr_experiments,
    'vector' : vector_experiments
}