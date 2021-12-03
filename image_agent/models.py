import torch
import torch.nn.functional as F

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)
    return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
            for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]
            
class Detector(torch.nn.Module):

    class BlockUpConv(torch.nn.Module):
        def __init__(self, c_in, c_out, stride=1, residual: bool = True):
            super().__init__()

            self.residual = residual
            self.upsample = None
            if stride != 1 or c_in != c_out:
                self.upsample = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c_in, c_out, kernel_size=1, stride=stride, output_padding=1,
                                             bias=False),
                    torch.nn.BatchNorm2d(c_out)
                )

            self.net = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(c_in, c_out, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(c_out),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(c_out, c_out, kernel_size=3, padding=1, stride=stride, output_padding=1,
                                         bias=False),
                torch.nn.BatchNorm2d(c_out),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(c_out, c_out, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(c_out),
                torch.nn.ReLU()
            )

        def forward(self, x):
            if self.residual:
                identity = x if self.upsample is None else self.upsample(x)
                return self.net(x) + identity
            else:
                return self.net(x)

    class BlockConv(torch.nn.Module):
        def __init__(self, c_in, c_out, kernel_size=3, stride=1, residual: bool = True):
            super().__init__()
            
            self.residual = residual
            self.downsample = None
            if stride != 1 or c_in != c_out:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(c_out)
                )

            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(c_out),
                torch.nn.ReLU(),
                torch.nn.Conv2d(c_out, c_out, kernel_size=kernel_size, padding=(kernel_size // 2), stride=stride,
                                bias=False),
                torch.nn.BatchNorm2d(c_out),
                torch.nn.ReLU(),
                torch.nn.Conv2d(c_out, c_out, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(c_out),
                torch.nn.ReLU()
            )

        def forward(self, x):
            if self.residual:
                identity = x if self.downsample is None else self.downsample(x)
                return self.net(x) + identity
            else:
                return self.net(x)

    def __init__(self, dim_layers=[32, 64, 128], c_in=3, c_out=2, input_normalization: bool = True,
                 skip_connections: bool = True, residual: bool = False):
        super().__init__()

        self.skip_connections = skip_connections

        c = dim_layers[0]
        self.net_conv = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c, kernel_size=7, padding=3, stride=2, bias=False),
            torch.nn.BatchNorm2d(c),
            torch.nn.ReLU()
        )])
        self.net_upconv = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(c * 2 if skip_connections else c, c_out, kernel_size=7,
                                     padding=3, stride=2, output_padding=1)
        ])
        for k in range(len(dim_layers)):
            l = dim_layers[k]
            self.net_conv.append(self.BlockConv(c, l, stride=2, residual=residual))
            l = l * 2 if skip_connections and k != len(dim_layers) - 1 else l
            self.net_upconv.insert(0, self.BlockUpConv(l, c, stride=2, residual=residual))
            c = dim_layers[k]

        if input_normalization:
            self.norm = torch.nn.BatchNorm2d(c_in)
        else:
            self.norm = None

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)

        h = x.shape[2]
        w = x.shape[3]

        skip_con = []
        for layers in self.net_conv:
            x = layers(x)
            skip_con.append(x)
        skip_con.pop(-1)
        skip = False
        for layers in self.net_upconv:
            if skip and len(skip_con) > 0:
                x = torch.cat([x, skip_con.pop(-1)], 1)
                x = layers(x)
            else:
                x = layers(x)
                skip = self.skip_connections

        pred = x[:, 0, :h, :w]
        boxes = x[:, 1, :h, :w]

        return pred, boxes

    def detect(self, image, max_pool_ks=7, min_score=0.2, max_det=15):
        heatmap, boxes = self(image[None])  
        heatmap = torch.sigmoid(heatmap.squeeze(0).squeeze(0)) 
        sizes = boxes.squeeze(0)
        return [(peak[0], peak[1], peak[2], (sizes[peak[2], peak[1]]).item())
                for peak in extract_peak(heatmap, max_pool_ks, min_score, max_det)]

def save_model(model, name: str = 'det.th'):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), name))


def load_model(name: str = 'det.th'):
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), name), map_location='cpu'))
    return r