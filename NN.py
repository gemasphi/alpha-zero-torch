import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class NetWrapper(object):
    def __init__(self, game, **params):
        super(NetWrapper, self).__init__()
        self.nn = AlphaZeroNet(game, params['n_res_layers'])
        self.optimizer = optim.Adam(self.nn.parameters(), lr = 0.2, weight_decay = 0.1)

    def train(self, data, batch_size = 32, loss_display = 2, epochs = 10):
        self.nn.train()
        data = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle=True)
        
        for epoch in range(epochs): 
            running_loss = 0.0
            for i, batch in enumerate(data, 0):
                board, policy, outcome = batch
                self.optimizer.zero_grad()
                v, p = self.nn(board.float())
                loss = self.nn.loss((v, p), (outcome, policy))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if i!= 0 and i % loss_display == 0:    
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / loss_display))
                    running_loss = 0.0

        return self.nn
    
    def predict(self, board):
        self.nn.eval()
        board = torch.Tensor(board)
        with torch.no_grad():
            v, p = self.nn(board)

        p = p.detach().numpy()
        return v, p

    def save_model(self, folder = "models", model_name = "model.pt"):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        torch.save({
            'model_state_dict': self.nn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, "{}/{}".format(folder, model_name))

    def load_model(self, path = "models/model.pt"):
        cp = torch.load(path)
        self.nn.load_state_dict(cp['model_state_dict'])
        self.optimizer.load_state_dict(cp['optimizer_state_dict'])

        return self.nn
        
class AlphaZeroNet(nn.Module):
    def __init__(self, game, res_layer_number = 5):
        super(AlphaZeroNet, self).__init__()

        input_planes = game.get_input_planes()
        board_dim = game.get_board_dimensions()

        self.conv = ConvLayer(board_dim = board_dim, inplanes = input_planes)
        self.res_layers = [ ResLayer() for i in range(res_layer_number)]
        self.valueHead = ValueHead(board_dim = board_dim)
        self.policyHead = PolicyHead(board_dim = board_dim, action_size = game.get_action_size(), output_planes = game.get_output_planes())

    def forward(self,s):
        s = self.conv(s)

        for res_layer in self.res_layers:
            s = res_layer(s)

        v = self.valueHead(s)
        p = self.policyHead(s)

        return v, p

    def loss(self, predicted, label):
        (v, p) = predicted
        (z, pi) = label

        value_error = (z.float() - torch.transpose(v,0,1))**2
        policy_error = (pi.float()*p.log()).sum(1)

        return (value_error - policy_error).mean() #no need to add the l2 regularization term as it is done in the optimizer

class ConvLayer(nn.Module):
    def __init__(self, board_dim = (), inplanes = 1, planes=128, stride=1):
        super(ConvLayer, self).__init__()
        self.inplanes = inplanes
        self.board_dim = board_dim
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, s):
        s = s.view(-1, self.inplanes, self.board_dim[0], self.board_dim[1])  # batch_size x planes x board_x x board_y
        s = F.relu(self.bn(self.conv(s)))

        return s

class ResLayer(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1):
        super(ResLayer, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        
        return out


class PolicyHead(nn.Module):
    def __init__(self, board_dim = (), action_size = -1, output_planes = -1):
        super(PolicyHead, self).__init__()
        self.board_dim = board_dim
        self.action_size = action_size
        self.output_planes = output_planes

        self.conv1 = nn.Conv2d(128, 32, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        if self.output_planes > 1:
            #TODO: this isnt being coverted into the action_size
            self.conv2 = nn.Conv2d(32, self.output_planes, kernel_size=1) # policy head
        else:
            self.fc = nn.Linear(self.board_dim[0]*self.board_dim[1]*32, self.action_size)

    def forward(self,s):
        p = F.relu(self.bn1(self.conv1(s))) # policy head

        if self.output_planes > 1:
            p = conv2(p)
        else:
            p = p.view(-1, self.board_dim[0]*self.board_dim[1]*32)
            p = self.fc(p)
            
        p = self.logsoftmax(p).exp()

        return p


class ValueHead(nn.Module):
    def __init__(self, board_dim = ()):
        super(ValueHead, self).__init__()
        self.board_dim = board_dim
        self.conv = nn.Conv2d(128, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(self.board_dim[0]*self.board_dim[1], 32) 
        self.fc2 = nn.Linear(32, 1)

    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, self.board_dim[0]*self.board_dim[1])  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        return v