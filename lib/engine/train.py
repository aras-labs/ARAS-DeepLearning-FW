import os
import torch
import torchvision
from tqdm import tqdm
from lib.engine.evaluation import evaluate


def train_on_batch(model, input_batch, target_batch, optimizer, loss_fn, metrics=None):
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        preds = model(input_batch)
        loss = loss_fn(preds, target_batch)
    loss.backward()
    optimizer.step()

    metrics_value = {}
    for m in metrics:
        metrics_value[m] = metrics[m](preds, target_batch).item()

    return loss, metrics_value, preds


def train_on_epoch(model, data_loader, epoch, optimizer,
                   loss_fn, summery_writer=None, metrics=None, device=None,
                   data_prepare_func=None, output_logger_func=None):
    model.train()

    running_metrics = {}
    for m in metrics:
        running_metrics[m] = 0
    running_loss = 0.0

    t = tqdm(data_loader, desc='Initialize Epoch {} of {} ..'.format(*epoch))
    for i, (input_batch, target_batch) in enumerate(t):
        # if data_prepare_func is not None:
        #     input_batch, target_batch = data_prepare_func(input_batch, target_batch)
        if isinstance(input_batch, dict):
            for k in input_batch.keys():
                input_batch[k] = input_batch[k].to(device)
        else:
            input_batch = input_batch.to(device)

        if isinstance(target_batch, dict):
            for k in target_batch.keys():
                target_batch[k] = target_batch[k].to(device)
        else:
            target_batch = target_batch.to(device)

        loss, metrics_value, model_output = train_on_batch(model, input_batch, target_batch, optimizer, loss_fn, metrics=metrics)
        running_loss += loss

        for m in metrics_value:
            running_metrics[m] += metrics_value[m]
            summery_writer.add_scalar('Metrics/{}'.format(m), metrics_value[m], len(data_loader) * epoch + i)
        if i % 10 == 0:
            t.set_description(
                "Epoch: {}/{}, Batch: {}/{}, Loss: {:.4}".format(epoch[0], epoch[1], i, len(data_loader), loss.item()))
            t.refresh()  # to show immediately the update
            summery_writer.add_scalar('batch Losses', loss, len(data_loader) * epoch[0] + i)
            if i % 1000 == 0:
                # TODO creat some function for other configuration
                if output_logger_func is not None:
                    output_logger_func(input_batch, target_batch, model_output)
                # img_grid = torchvision.utils.make_grid(torch.unsqueeze(torch.argmax(model_output, 1), 1))
                # print(img_grid.shape)
                # summery_writer.add_image('model output Epoch/Batch: {}/{}'.format(epoch, i), img_grid)

                # img_grid = torchvision.utils.make_grid(input_batch)
                # summery_writer.add_image('input image Epoch/Batch: {}/{}'.format(epoch, i), img_grid)

                # img_grid = torchvision.utils.make_grid(target_batch)
                # summery_writer.add_image('target image Epoch/Batch: {}/{}'.format(epoch, i), img_grid)

                for m in metrics_value:
                    summery_writer.add_scalar('Metrics/{}'.format(m), metrics_value[m], len(data_loader) * epoch + i)
            summery_writer.flush()

    epoch_loss = running_loss / len(data_loader)
    return epoch_loss


def train(model, data_loader, epochs, optimizer,
          loss_fn, weight_folder=None, summery_writer=None, metrics=None,
          device=None, val_data_loader=None, data_prepare_func=None, output_logger_func=None):

    if metrics is None:
        metrics = []

    model.to(device)
    if weight_folder is not None:
        torch.save(model.state_dict(), os.path.join(weight_folder,
                                                    'init_weights'))
    for epoch in range(epochs):

        epoch_loss = train_on_epoch(model, data_loader, (epoch, epochs), optimizer,
                                    loss_fn, summery_writer=summery_writer, metrics=metrics,
                                    device=device, data_prepare_func=data_prepare_func,
                                    output_logger_func=output_logger_func)

        validation_results = evaluate(model, val_data_loader, metrics={'loss': loss_fn},
                                      device=device, data_prepare_func=data_prepare_func)

        summery_writer.add_scalars('Epoch Losses', {'Train': epoch_loss, 'Val': validation_results['loss']}, epoch)
        summery_writer.flush()
        if weight_folder is not None:
            torch.save(model.state_dict(), os.path.join(weight_folder,
                                                        'epoch_{}.ckp'.format(epoch)))

