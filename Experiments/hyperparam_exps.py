from PatientVec.models.Vanilla import ClassificationTrainer as BasicCT
from PatientVec.models.Hierarchical import ClassificationTrainer as HierCT
from PatientVec.trainer import Trainer, Evaluator
from PatientVec.Experiments.modifiable_config_exp import vanilla_configs, attention_configs, hierarchical_configs, structured_configs

def get_basic_data(data, truncate=90, encodings=None) :
    train_data = data.filter_data_length(data.get_data('train', structured=True, encodings=encodings), truncate=truncate)
    dev_data = data.filter_data_length(data.get_data('dev', structured=True, encodings=encodings), truncate=truncate)

    return train_data, dev_data
    
def vanilla_experiments(data, args) :
    train_data, dev_data = get_basic_data(data, truncate=90)

    for e in vanilla_configs :
        config = e(data, structured=True, args=args)
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
    train_data, dev_data = get_basic_data(data, truncate=90)

    for e in attention_configs :
        config = e(data, structured=True, args=args)
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
    train_data, dev_data = get_basic_data(data, truncate=90)

    for e in hierarchical_configs :
        config = e(data, structured=True, args=args)
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
    train_data, dev_data = get_basic_data(data, truncate=90, encodings=data.structured_columns)

    for e in structured_configs :
        config = e(data, structured=True, encodings=data.structured_columns, args=args)
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

experiment_types = {
    'vanilla' : vanilla_experiments,
    'attention' : attention_experiments,
    'hierarchical' : hierarchical_experiments,
    'structured' : structured_attention_experiments
}