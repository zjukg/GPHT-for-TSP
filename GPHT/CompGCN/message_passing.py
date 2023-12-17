import inspect

import torch
from torch_scatter import scatter


def scatter_(name, src, index, dim_size=None):
    if name == 'add':
        name = 'sum'
    assert name in ['sum', 'mean', 'max']
    out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
    return out[0] if isinstance(out, tuple) else out


class MessagePassing(torch.nn.Module):
    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()

        self.message_args = inspect.getargspec(self.message)[0][1:]
        self.update_args = inspect.getargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])

        update_args = [kwargs[arg] for arg in self.update_args]

        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0], dim_size=size)
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out
