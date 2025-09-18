class ObjectView:
    def __init__(self, d):
        self.__dict__ = d


args = {
    'n_interactions': 2,
    'n_filters': 64,
    'output_dim': 1,
    'batch_size': 8,
    'epochs': 500,
    'opt': 'adam',
    'opt_scheduler': 'step',
    'opt_decay_step': 10,
    'opt_decay_rate': 0.96,
    'weight_decay': 1e-4,
    'lr': 1e-3,
    'es_patience': 40
}

args = ObjectView(args)

