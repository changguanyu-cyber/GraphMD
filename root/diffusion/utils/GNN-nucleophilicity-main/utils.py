import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


def build_optimizer(args, params):
    weight_decay = args.weight_decay

    # return an iterator
    filter_fn = filter(lambda p: p.requires_grad, params)  # params is an generator (kind of iterator)

    # optimizer
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)

    # scheduler
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'reduceOnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         patience=args.lr_decay_patience,
                                                         factor=args.lr_decay_factor)
    else:
        raise Exception('Unknown optimizer type')

    return scheduler, optimizer


def standardize_cdft(train_data, val_data):
    # extract cdft
    train_x = [data.cdft for data in train_data]
    train_x = torch.cat(train_x, dim=0).cpu()
    val_x = [data.cdft for data in val_data]
    val_x = torch.cat(val_x, dim=0).cpu()
    # standardize
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = torch.tensor(scaler.transform(train_x), dtype=torch.float32)
    val_x = torch.tensor(scaler.transform(val_x), dtype=torch.float32)
    for i in range(train_x.size(0)):
        train_data[i].cdft = train_x[[i]]
    for j in range(val_x.size(0)):
        val_data[j].cdft = val_x[[j]]


