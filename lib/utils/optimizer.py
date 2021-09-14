import torch.optim as optim


def get_optimizer(params, optimizer, args):
    return eval('optim.{}'.format(optimizer))(params, **dict(args))

