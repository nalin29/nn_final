import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class ResNetBlock(torch.nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(c_in)
        self.c1 = torch.nn.Conv2d(
            c_in, c_out, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn2 = torch.nn.BatchNorm2d(c_out)
        self.c2 = torch.nn.Conv2d(
            c_out, c_out, kernel_size, padding=kernel_size//2)
        self.identity = torch.nn.Identity()
        if c_in != c_out or stride != 1:
            self.identity = torch.nn.Sequential(
                torch.nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(c_out)
            )

    def forward(self, x):
        x_in = x
        x = self.bn1(x)
        x = self.c1(torch.relu(x))
        x = self.bn2(x)
        x = self.c2(torch.relu(x))
        return x + self.identity(x_in)


class Planner(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        self.bn = torch.nn.BatchNorm2d(3)
        features = [32, 64, 128]
        self.ups = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        in_channels = 3

        # Encode
        for feature in features:
            self.downs.append(torch.nn.Sequential(
                ResNetBlock(in_channels, feature)))
            in_channels = feature

        # Decode
        for feature in reversed(features):
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.BatchNorm2d(feature*2),
                    torch.nn.ConvTranspose2d(
                        feature*2, feature, kernel_size=2, stride=2,)
                )
            )
            self.ups.append(torch.nn.Sequential(
                ResNetBlock(feature*2, feature)))
        self.bottleneck = torch.nn.Sequential(
            ResNetBlock(features[-1], features[-1]*2))
        self.final_conv = torch.nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        input = img
        x = self.bn(input)

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                skip_connection = skip_connection[:,
                                                  :, :x.shape[2], :x.shape[3]]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        heat = self.final_conv(x)[:, :, :input.shape[2], :input.shape[3]]
        return heat.view(heat.shape[0], heat.shape[2], heat.shape[3])

    def predict_point(self, img):
        heat = self(img)
        return spatial_argmax(heat)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(
        path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r
