import argparse
from d8.image_classification import Dataset
import autogluon.core as ag
from gluoncv.auto.tasks import ImageClassification
import logging
debug_log_file = 'searcher_debug.log'
fileHandler = logging.FileHandler(debug_log_file)
logging.getLogger('autogluon').addHandler(fileHandler)

"""list
['ibeans', 'boat', 'intel', 'fruits-360', 
'caltech-256', 'cub-200', 'cifar10', 'citrus-leaves', 
'cmaterdb', 'cassava', 'dtd', 'eurosat', 'food-101', 
'horses-or-humans', 'malaria', 'flower-102', 'green-finder', 
'leaves', 'plant-village', 'rock-paper-scissors', 'sun-397', 
'chessman', 'casting-products', 'monkey-10', 'dog-cat-panda', 
'broad-leaved-dock', 'food-or-not-food', 'gemstones', 'hurricane-damage', 
'animal-10', 'walk-or-run', 'gender', 'brain-tumor', 'facial-expression', 
'rice-diseases', 'mushrooms', 'oregon-wildlife', 'bird-225', 'stanford-dogs', 
'butterfly', 'dogs-vs-cats', 'deep-weeds', 'oxford-pets', 'lego-brick', 
'satelite-plane', 'honey-bee', 'coil-100', 'flower-10']
"""

_PRESETS = {
    'default': {
        'model': ag.Categorical('mobilenetv2_0.25', 'mobilenetv3_small', 'mobilenetv3_large', 
                                'resnet18_v1b', 'resnet34_v1b', 'resnet50_v1b', 'resnet101_v1b',
                                'vgg16_bn', 'SE_ResNext50_32x4d', 'resnest50', 'resnest200'),
        'num_trials': 25,
        'lr': ag.Real(1e-4, 1e-2, log=True),
        'batch_size': ag.Int(3, 8),  # min: (2 ** 3) = 8, max:  (2 ** 8) = 512
        'exp_batch_size': True,
        'ngpus_per_trial': 4,
        'nthreads_per_trial': 64,
        'epochs': 200,
        'search_strategy': 'bayesopt',
        'search_options': {'debug_log': True, 'base_estimator': 'RF', 'acq_func': 'EI'},
        'max_reward': 1.0
    }
}


def parse_args():
    parser = argparse.ArgumentParser('Image Classification HPO')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--num-trials', type=int, default=-1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=-1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.name in _PRESETS:
        config = _PRESETS[args.name]
    else:
        config = _PRESETS['default']
    
    if args.num_trials >= 1:
        config['num_trials'] = args.num_trials
    
    if args.optimizer:
        config['custom_optimizer'] = args.optimizer

    if args.epochs >= 1:
        config['epochs'] = args.epochs

    ds = Dataset.get(args.name)
    train, valid = ds.split(0.9, seed=0)

    task = ImageClassification(config)

    classifier = task.fit(train, valid)

    with open('fit_summary.txt', 'w') as f:
        print(task.fit_summary(), file=f)

    with open('fit_history.txt', 'w') as f:
        print(task.fit_history(), file=f)
    
    classifier.save('best_model.pkl')
    

    


