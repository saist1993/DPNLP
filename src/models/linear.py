# linear layer
import torch
import config
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Function


def laplace(epsilon, L1_norm):
    b = L1_norm / epsilon
    return b

def initialize_parameters(m):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.05, 0.05)
    elif isinstance(m, nn.LSTM):
        for n, p in m.named_parameters():
            if 'weight_ih' in n:
                i, f, g, o = p.chunk(4)
                nn.init.xavier_uniform_(i)
                nn.init.xavier_uniform_(f)
                nn.init.xavier_uniform_(g)
                nn.init.xavier_uniform_(o)
            elif 'weight_hh' in n:
                i, f, g, o = p.chunk(4)
                nn.init.orthogonal_(i)
                nn.init.orthogonal_(f)
                nn.init.orthogonal_(g)
                nn.init.orthogonal_(o)
            elif 'bias' in n:
                i, f, g, o = p.chunk(4)
                nn.init.zeros_(i)
                nn.init.ones_(f)
                nn.init.zeros_(g)
                nn.init.zeros_(o)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        try:
            nn.init.zeros_(m.bias)
        except AttributeError:
            pass



class GradReverse(Function):
    """
        Torch function used to invert the sign of gradients (to be used for argmax instead of argmin)
        Usage:
            x = GradReverse.apply(x) where x is a tensor with grads.

        Copied from here: https://github.com/geraltofrivia/mytorch/blob/0ce7b23ff5381803698f6ca25bad1783d21afd1f/src/mytorch/utils/goodies.py#L39
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class GradReverse(Function):
    """
        Torch function used to invert the sign of gradients (to be used for argmax instead of argmin)
        Usage:
            x = GradReverse.apply(x) where x is a tensor with grads.

        Copied from here: https://github.com/geraltofrivia/mytorch/blob/0ce7b23ff5381803698f6ca25bad1783d21afd1f/src/mytorch/utils/goodies.py#L39
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class Linear(nn.Module):

    def __init__(self, model_params):
        super().__init__()
        number_of_layers, input_dim, hidden_dim, output_dim, dropout = \
            model_params['number_of_layers'], model_params['input_dim'],\
            model_params['hidden_dim'], model_params['output_dim'], model_params['dropout']

        if type(hidden_dim) == list:
            assert len(hidden_dim) + 1 == number_of_layers
        else:
            hidden_dim = [hidden_dim for i in range(number_of_layers-1)]


        '''
            hidden_dims = 1000, 500, 100
            
            Layer1: input_dim - 1000 dim
            Layer2: 1000 dim -  500 dim
            Layer3: 500 dim - 100 dim
            Layer4: 100 dim - output dim
            
            Thus no. of layers == hidden dim + 1
        
        '''

        self.fc_layers = []
        self.dropout = nn.Dropout(dropout)

        for i in range(number_of_layers):
            if i != number_of_layers - 1 and i != 0:
                self.fc_layers.append((nn.Linear(hidden_dim[i-1], hidden_dim[i])))
            elif i == 0:
                self.fc_layers.append(nn.Linear(input_dim, hidden_dim[i]))
            else:
                self.fc_layers.append(nn.Linear(hidden_dim[i-1], output_dim, bias=True)) # @TODO: see if there is a need for a softmax via sigmoid or something

        self.fc_layers = nn.ModuleList(self.fc_layers)

        self.noise_layer = False

    def forward(self, params):
        length = None # it is a dummy input only meant for legacy
        x = params['input']
        for index, layer in enumerate(self.fc_layers):
            if len(self.fc_layers)-1 != index:
                x = self.dropout(func.relu(layer(x)))
                # x = func.relu(layer(x))
            else:
                x = layer(x)
        return x

    @property
    def layers(self):
        return nn.ModuleList(
            [self.fc_layers])


class LinearAdv(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.encoder = Linear(params['model_arch']['encoder'])
        self.classifier = Linear(params['model_arch']['main_task_classifier'])
        self.adv = Linear(params['model_arch']['adv'])
        self.noise_layer = params['noise_layer']
        self.adv.apply(initialize_parameters)  # don't know, if this is needed.
        self.classifier.apply(initialize_parameters)  # don't know, if this is needed.
        self.encoder.apply(initialize_parameters)  # don't know, if this is needed.
        self.eps = params['eps']
        self.device = params['device']


    def forward(self, params):

        text, gradient_reversal = \
            params['input'], params['gradient_reversal']

        original_hidden = self.encoder(params)

        if self.noise_layer:
            m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([laplace(self.eps, 2)]))
            hidden = original_hidden/torch.norm(original_hidden, keepdim=True, dim=1)
            hidden = hidden + m.sample(hidden.shape).squeeze().to(self.device)
        else:
            hidden = original_hidden

        _params = {}
        # classifier setup
        _params['input'] = hidden
        prediction = self.classifier(_params)


        # adversarial setup
        if gradient_reversal:
            _params['input'] = GradReverse.apply(hidden)
        else:
            _params['input'] = hidden
        adv_output = self.adv(_params)

        #
        # if return_hidden:
        #     return prediction, adv_output, original_hidden, hidden

        output = {
            'prediction': prediction,
            'adv_output': adv_output
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.encoder, self.classifier, self.adv])

if __name__ == '__main__':
    #
    # model_params = {
    #     'number_of_layers': 4,
    #     'hidden_dim': 100,
    #     'output_dim': 50,
    #     'dropout': 0.5,
    #     'input_dim': 200
    # }
    #
    # lin = Linear(model_params)
    # print(lin)
    #
    #
    # model_params = {
    #     'number_of_layers': 4,
    #     'hidden_dim': [1000, 500, 100],
    #     'output_dim': 50,
    #     'dropout': 0.5,
    #     'input_dim': 5000
    # }
    #
    # lin = Linear(model_params)
    # print(lin)
    #
    #
    # model_params = {
    #     'number_of_layers': 2,
    #     'hidden_dim': [100],
    #     'output_dim': 50,
    #     'dropout': 0.5,
    #     'input_dim': 5000
    # }
    #
    # lin = Linear(model_params)
    # print(lin)
    model_arch = config.amazon_model
    model_arch['encoder']['input_dim'] = 5000
    model_arch['main_task_classifier']['output_dim'] = 2
    model_arch['adv']['output_dim'] = 2
    model_params = {
        'model_arch': config.amazon_model,
        'noise_layer': True,
        'eps': 10.0,
        'device': 'cpu'
    }
    la = LinearAdv(model_params)
    print(la)


