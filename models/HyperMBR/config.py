import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': ([0.001], 'learning rate'),
        'dropout': ([0.3], 'dropout probability'),
        'feat_dim':(64,'feature dimension'),
        'embed_L2_norm':(1e-7,'embed_L2_norm'),
        'negative_num':([256],'the number of negative samples for each user-bundle_p pair'),
        'create_embeddings':(True,'whether initial the feature embedding'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (200, 'maximum number of epochs to train for'),
        'early':(50,'early stop'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (125, 'seed for training'),
        'log-freq': (5, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (30, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs'),
        'dkd_alpha': (1.0, 'dkd alpha'),
        'dkd_beta': (8.0, 'dkd beta'),
        'dkd_temperature': (2, 'dkd temperature'),
        'use_distillation':(True,'whether to use distillation'),
        'warmup':(20,'DKD warn up epoch'),
        'itemLevelCoef':(0.7,'itemLevel KD coef'),
        'bundleLevelCoef':(0.3,'bundelLevel KD coef'),
        'temperature': (1.0, 'Temperature of distillation')
    },
    'model_config': {
        'task': ('brec', 'which tasks to train on, can be any of [lp, nc,brec]'),
        'model': ('AdaHypBR', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HyperGCN]'),
        'dim': ([64], 'embedding dimension'),
        'manifold': (['Hyperboloid'], 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'itemLevel_c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'bundleLevel_c': (2.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': ([2], 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (relu,leaky_relu,or None for no activation)'),
        'n-heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not'),
        'concat':(False,'')
    },
    'data_config': {
        # 'dataPath': ('../input/hbgcn-1/data', 'path of dataset'),
        'dataPath': ('/kaggle/input/hyper-ori-youshu/data', 'path of dataset'),
        'log': ('/kaggle/working/log', 'log save path '),
        'recommendation_result_interval':(40,''),
        'sample':('simple', 'which sample way to use'),
        'hard_window':([0.7, 1.0], 'which sample way to use'),  # top 30%
        'hard_prob': ([0.1,0.1], 'which sample way to use'),  # probability 0.8
        'conti_train': ('Youshu-simple.pth', 'which sample way to use'),
        'dataset': ('Youshu', 'which dataset to use'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (123, 'seed for data splits (train/test/val)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
