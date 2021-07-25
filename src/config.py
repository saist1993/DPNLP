# dataset location
dataset_location = {
    'amazon': ['../data/amazon/', '/home/gmaheshwari/storage/amazon/', '../datasets/amazon/']
}

# model characterstics

amazon_model = {  # this will need more work!!
    'name': 'amazon',  # focuses on linear models in this case.
    'model_type': 'linear_adv',
    'hidden_dim': [1000, 500, 100],  # 256
    'dropout': 0.2,
    'n_layers': 2,
    'adv_number_of_layers': 2,
    'adv_dropout': 0.2,
    'num_filters': 100,
    'filter_sizes': [3, 4, 5]
}

amazon_model = {  # this will need more work!!

    'name': 'amazon',  # focuses on linear models in this case.

    'encoder': {
        'hidden_dim': [1000, 500],
        'output_dim': 100,
        'number_of_layers': 3,  # needs to be len(hidden) + 1
        'dropout': 0.5,
        'input_dim': -1  # This needs to be set
    },

    'main_task_classifier': {
        'hidden_dim': [100, 50],
        'output_dim': -1,  # This needs to be set
        'number_of_layers': 3,  # needs to be len(hidden) + 1
        'dropout': 0.5,
        'input_dim': 100  # This needs to be set
    },

    'adv': {
        'hidden_dim': [100],
        'output_dim': -1,  # This needs to be set
        'number_of_layers': 2,  # needs to be len(hidden) + 1
        'dropout': 0.4,
        'input_dim': 100  # This needs to be set
    }
    #
    # 'model_type': 'linear_adv',
    # 'hidden_dim': [1000, 500, 100],      # 256
    # 'dropout': 0.2,
    # 'n_layers': 2,
    # 'adv_number_of_layers': 2,
    # 'adv_dropout' : 0.2,
    # 'num_filters': 100,
    # 'filter_sizes': [3,4,5]
}
