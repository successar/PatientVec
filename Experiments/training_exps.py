from PatientVec.models.Vanilla import ClassificationTrainer as BasicCT
from PatientVec.models.Hierarchical import ClassificationTrainer as HierCT
from PatientVec.trainer import Trainer, Evaluator
from PatientVec.Experiments.config_exp import basic_configs, hierarchical_configs, structured_configs

def get_basic_data(data, truncate=90, encodings=None) :
    train_data = data.filter_data_length(data.get_data('train', structured=True, encodings=encodings), truncate=truncate)
    dev_data = data.filter_data_length(data.get_data('dev', structured=True, encodings=encodings), truncate=truncate)

    return train_data, dev_data
    
def basic_experiments(data, args) :
    train_data, dev_data = get_basic_data(data, truncate=90)

    for e in basic_configs :
        for use_structured in [True, False] :
            config = e(data, structured=use_structured)
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
        for use_structured in [True, False] :
            config = e(data, structured=use_structured)
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
        for use_structured in [True, False] :
            config = e(data, structured=use_structured, encodings=data.structured_columns)
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