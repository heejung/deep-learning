import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.2):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, features, captions):
        batch_size = captions.size(0)
        x = self.embed(captions[:,:-1])
        x = torch.cat((features.view(x.size(0), 1, -1), x), 1)
        x, hidden = self.lstm(x)
        x = x.contiguous().view(-1, self.hidden_size)
        x = self.fc(self.dropout(x))
        x = x.view(batch_size, -1, self.vocab_size)
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        seq = []
        x = inputs
        if states is None:
            hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                      torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        else:
            hidden = states
        for _ in range(max_len):
            x, hidden = self.lstm(x, hidden)
            x = x.contiguous().view(-1, self.hidden_size)
            x = self.fc(x)
            x = x.squeeze()
            x = torch.argmax(x)
            seq.append(x.item())
            x = x.view(1, -1)
            x = self.embed(x)
        return seq