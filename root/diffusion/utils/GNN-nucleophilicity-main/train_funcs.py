import torch
import torch.nn.functional as F
import copy
import time


def train(model, loaders: list, optimizer, scheduler, args):

    history = {'train_loss': [], 'val_loss': []}
    train_loader = loaders[0]
    val_loader = loaders[1]

    best_val_loss = 1e9
    best_model = model
    es = 0
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        train_epoch_loss = 0.0
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(args.device)
            optimizer.zero_grad()

            pred = model(batch)
            label = batch.y

            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        train_epoch_loss /= len(train_loader)

        # ---validation---
        val_epoch_loss, val_epoch_mae, val_epoch_rmse, _, _, _ = test(model, val_loader, args)
        model.train()
        if args.opt_scheduler is None:
            pass
        elif args.opt_scheduler == 'reduceOnPlateau':
            scheduler.step(val_epoch_loss)
        elif args.opt_scheduler == 'step':
            scheduler.step()

        # record epoch_loss
        history['train_loss'].append(train_epoch_loss)
        history['val_loss'].append(val_epoch_loss)

        # print training process
        log = 'Epoch: {:03d}/{:03d}; ' \
              'AVG Training Loss (MSE):{:.8f}; ' \
              'AVG Val Loss (MSE):{:.8f};' \
              'lr:{:8f}'
        print(time.strftime('%H:%M:%S'), log.format(epoch+1, args.epochs,
                                                    train_epoch_loss,
                                                    val_epoch_loss,
                                                    current_lr))
        if current_lr != optimizer.param_groups[0]['lr']:
            print('lr has been updated from {:.8f} to {:.8f}'.format(current_lr,
                                                                     optimizer.param_groups[0]['lr']))

        # determine whether stop early by val_epoch_loss
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model = copy.deepcopy(model)
            es = 0
        else:
            es += 1
            print("Counter {} of patience {}".format(es, args.es_patience))
            if es > args.es_patience:
                print("Early stopping with best_val_loss {:.8f}".format(best_val_loss))
                break

    return history, best_model


def test(model, loader, args):
    model.eval()

    mol_ls = []
    pred_ls = []
    y_ls = []
    for batch in loader:
        batch = batch.to(args.device)
        with torch.no_grad():
            #
            mol_ls += batch.name
            pred_ls.append(model(batch))
            y_ls.append(batch.y)

    pred = torch.cat(pred_ls, dim=0).reshape(-1)
    y = torch.cat(y_ls, dim=0).reshape(-1)

    test_loss = model.loss(pred, y)
    mae = F.l1_loss(pred.reshape(-1), y.reshape(-1))
    rmse = torch.sqrt(test_loss)

    return test_loss, mae, rmse, mol_ls, y, pred

