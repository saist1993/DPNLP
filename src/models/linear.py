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
            model_params['number_of_layers'], model_params['input_dim'], \
            model_params['hidden_dim'], model_params['output_dim'], model_params['dropout']

        if type(hidden_dim) == list:
            assert len(hidden_dim) + 1 == number_of_layers
        else:
            hidden_dim = [hidden_dim for i in range(number_of_layers - 1)]

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

        if number_of_layers == 1:
            self.fc_layers.append(nn.Linear(input_dim, output_dim,
                                            bias=True))
        else:
            for i in range(number_of_layers):
                if i != number_of_layers - 1 and i != 0:
                    self.fc_layers.append((nn.Linear(hidden_dim[i - 1], hidden_dim[i])))
                elif i == 0:
                    self.fc_layers.append(nn.Linear(input_dim, hidden_dim[i]))
                else:
                    self.fc_layers.append(nn.Linear(hidden_dim[i - 1], output_dim,
                                                    bias=True))  # @TODO: see if there is a need for a softmax via sigmoid or something

        self.fc_layers = nn.ModuleList(self.fc_layers)

        self.noise_layer = False

    def forward(self, params):
        length = None  # it is a dummy input only meant for legacy
        x = params['input']
        all_outputs = []
        for index, layer in enumerate(self.fc_layers):
            if len(self.fc_layers) - 1 != index:
                x = self.dropout(func.relu(layer(x)))
                # x = func.relu(layer(x))
                all_outputs.append(x)
            else:
                x = layer(x)
                all_outputs.append(x)
        return x, all_outputs

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
        # self.second_adv = Linear(params['model_arch']['adv'])
        self.noise_layer = params['noise_layer']
        self.adv.apply(initialize_parameters)  # don't know, if this is needed.
        self.classifier.apply(initialize_parameters)  # don't know, if this is needed.
        self.encoder.apply(initialize_parameters)  # don't know, if this is needed.
        # self.second_adv.apply(initialize_parameters)
        self.eps = params['eps']
        self.device = params['device']
        self.apply_noise_to_adv = params['apply_noise_to_adv']

    def forward(self, params):

        text, gradient_reversal = \
            params['input'], params['gradient_reversal']

        original_hidden, encoder_other_layer_output = self.encoder(params)
        copy_original_hidden = original_hidden.clone().detach()

        if self.noise_layer:
            m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([laplace(self.eps, 2)]))
            hidden = original_hidden / torch.norm(original_hidden, keepdim=True, dim=1)
            hidden = hidden + m.sample(hidden.shape).squeeze().to(self.device)
        else:
            hidden = original_hidden

        _params = {}
        # classifier setup
        _params['input'] = hidden
        prediction, classifier_hiddens = self.classifier(_params)

        # new_hidden = torch.cat((hidden, copy_original_hidden),1)
        # adversarial setup
        if self.apply_noise_to_adv:
            if gradient_reversal:
                _params['input'] = GradReverse.apply(hidden)
            else:
                _params['input'] = hidden
        else:
            if gradient_reversal:
                _params['input'] = GradReverse.apply(copy_original_hidden)
            else:
                _params['input'] = copy_original_hidden

        adv_output, adv_hiddens = self.adv(_params)
        # _params['input'] = GradReverse.apply(copy_original_hidden)
        # second_adv_output = self.second_adv(_params)

        #
        # if return_hidden:
        #     return prediction, adv_output, original_hidden, hidden

        output = {
            'prediction': prediction,
            'adv_output': adv_output,
            'hidden': hidden,
            'classifier_hiddens': classifier_hiddens,
            'adv_hiddens': adv_hiddens
            # 'second_adv_output': second_adv_output

        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.encoder, self.classifier, self.adv])

    def reset(self):
        self.adv.apply(initialize_parameters)  # don't know, if this is needed.
        self.classifier.apply(initialize_parameters)  # don't know, if this is needed.
        self.encoder.apply(initialize_parameters)  # don't know, if this is needed.

class SimpleLinear(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_dim = params['model_arch']['encoder']['input_dim']
        output_dim = params['model_arch']['encoder']['output_dim']
        self.encoder = nn.Linear(input_dim, output_dim)
        self.encoder.apply(initialize_parameters)

    def forward(self, params):
        text = params['input']
        prediction = self.encoder(text)

        output = {
            'prediction': prediction,
            'adv_output': None,
            'hidden': prediction, # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
            # 'second_adv_output': second_adv_output

        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.encoder])

    def reset(self):
        self.encoder.apply(initialize_parameters)  # don't know, if this is needed.

class SimpleNonLinear(nn.Module):
    def __init__(self, params):
        super().__init__()

        input_dim = params['model_arch']['encoder']['input_dim']
        output_dim = params['model_arch']['encoder']['output_dim']

        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(32)

    def forward(self, params):
        x = params['input']
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        output = {
            'prediction': x,
            'adv_output': None,
            'hidden': x,  # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
            # 'second_adv_output': second_adv_output

        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2, self.layer_3, self.layer_out])

    def reset(self):
        self.layer_1.apply(initialize_parameters)  # don't know, if this is needed.
        self.layer_2.apply(initialize_parameters)  # don't know, if this is needed.
        self.layer_3.apply(initialize_parameters)  # don't know, if this is needed.
        self.layer_out.apply(initialize_parameters)  # don't know, if this is needed.

class LinearAdvEncodedEmoji(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.encoder = Linear(params['model_arch']['encoder'])
        self.classifier = Linear(params['model_arch']['main_task_classifier'])
        self.adv_1 = Linear(params['model_arch']['adv'])
        self.adv_2 = Linear(params['model_arch']['adv'])
        self.adv_3 = Linear(params['model_arch']['adv'])

        # self.second_adv = Linear(params['model_arch']['adv'])
        self.noise_layer = params['noise_layer']
        self.adv_1.apply(initialize_parameters)  # don't know, if this is needed.
        self.adv_2.apply(initialize_parameters)  # don't know, if this is needed.
        self.adv_3.apply(initialize_parameters)  # don't know, if this is needed.
        self.classifier.apply(initialize_parameters)  # don't know, if this is needed.
        self.encoder.apply(initialize_parameters)  # don't know, if this is needed.

        self.eps = params['eps']
        self.device = params['device']
        self.apply_noise_to_adv = params['apply_noise_to_adv']

    def forward(self, params):

        text, gradient_reversal = \
            params['input'], params['gradient_reversal']

        original_hidden, encoder_other_layer_output = self.encoder(params)
        # original_hidden = func.tanh(original_hidden)
        # add tanh here!
        # copy_original_hidden = original_hidden.clone().detach()

        if self.noise_layer:
            m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([laplace(self.eps, 2)]))
            hidden = original_hidden / torch.norm(original_hidden, keepdim=True, dim=1)
            hidden = hidden + m.sample(hidden.shape).squeeze().to(self.device)
        else:
            hidden = original_hidden

        _params = {}
        # classifier setup
        _params['input'] = hidden
        prediction, prediction_other_layer_output = self.classifier(_params)

        # new_hidden = torch.cat((hidden, copy_original_hidden),1)
        # adversarial setup
        if gradient_reversal:
            _params['input'] = GradReverse.apply(hidden)
        else:
            _params['input'] = hidden

        adv_output_1, adv_other_layer_output_1  = self.adv_1(_params)
        adv_output_2, adv_other_layer_output_2 = self.adv_2(_params)
        adv_output_3, adv_other_layer_output_3 = self.adv_3(_params)
        # _params['input'] = GradReverse.apply(copy_original_hidden)
        # second_adv_output = self.second_adv(_params)

        #
        # if return_hidden:
        #     return prediction, adv_output, original_hidden, hidden

        output = {
            'prediction': prediction,
            # 'adv_output': adv_output,
            'hidden': hidden,
            'adv_output_1': adv_output_1,
            'adv_output_2': adv_output_2,
            'adv_output_3': adv_output_3,
            'adv_other_layer_output_1': adv_other_layer_output_1,
            'adv_other_layer_output_2': adv_other_layer_output_2,
            'adv_other_layer_output_3': adv_other_layer_output_3
            # 'second_adv_output': second_adv_output

        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.encoder, self.classifier, self.adv_1, self.adv_2, self.adv_3 ])

    def reset(self):
        self.adv_1.apply(initialize_parameters)  # don't know, if this is needed.
        self.adv_2.apply(initialize_parameters)  # don't know, if this is needed.
        self.adv_3.apply(initialize_parameters)  # don't know, if this is needed.
        self.classifier.apply(initialize_parameters)  # don't know, if this is needed.
        self.encoder.apply(initialize_parameters)  # don't know, if this is needed.


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


