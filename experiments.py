
import pdb
import sys


def run_experiments(dataset, args):

    if dataset == 'OAI':
        from OAI.train import (
            train_X_to_C,
            train_oracle_C_to_y_and_test_on_Chat,
            train_Chat_to_y_and_test_on_Chat,
            train_X_to_C_to_y,
            train_X_to_y,
            train_X_to_Cy,
            train_probe,
            test_time_intervention,
            hyperparameter_optimization
        )

    elif dataset == 'CUB':
        from CUB.train import (
            train_X_to_C,
            train_oracle_C_to_y_and_test_on_Chat,
            train_Chat_to_y_and_test_on_Chat,
            train_X_to_C_to_y,
            train_X_to_y,
            train_X_to_Cy,
            train_probe,
            test_time_intervention,
            robustness,
            hyperparameter_optimization
        )

    experiment = args[0].exp
    if experiment == 'Concept_XtoC':
        train_X_to_C(*args)

    elif experiment == 'Independent_CtoY':
        train_oracle_C_to_y_and_test_on_Chat(*args)

    elif experiment == 'Sequential_CtoY':
        train_Chat_to_y_and_test_on_Chat(*args)

    elif experiment == 'Joint':
        train_X_to_C_to_y(*args)

    elif experiment == 'Standard':
        train_X_to_y(*args)

    elif experiment == 'Multitask':
        train_X_to_Cy(*args)

    elif experiment == 'Probe':
        train_probe(*args)

    elif experiment == 'TTI':
        test_time_intervention(*args)

    elif experiment == 'Robustness':
        robustness(*args)

    elif experiment == 'HyperparameterSearch':
        hyperparameter_optimization(*args)

def parse_arguments():
    # First arg must be dataset, and based on which dataset it is, we will parse arguments accordingly
    assert len(sys.argv) > 2, 'You need to specify dataset and experiment'
    assert sys.argv[1].upper() in ['OAI', 'CUB'], 'Please specify OAI or CUB dataset'
    assert sys.argv[2].upper() in ['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                                   'Standard', 'Multitask', 'Joint', 'Probe',
                                   'TTI', 'Robustness', 'HyperparameterSearch'], 'Please specify valid experiment'
    dataset = sys.argv[1].upper()
    experiment = sys.argv[2].upper()

    # Handle accordingly to dataset
    if dataset == 'OAI':
        from OAI.train import parse_arguments
    elif dataset == 'CUB':
        from CUB.train import parse_arguments

    args = parse_arguments(experiment=experiment)
    return dataset, args

if __name__ == '__main__':
    run_experiments(*parse_arguments())
