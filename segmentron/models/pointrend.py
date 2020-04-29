import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from torchvision.models._utils import IntermediateLayerGetter
from .model_zoo import MODEL_REGISTRY
from .segbase import SegBaseModel
from ..config import cfg


@MODEL_REGISTRY.register(name='PointRend')
class PointRend(SegBaseModel):
    def __init__(self):
        super(PointRend, self).__init__(need_backbone=False)
        model_name = cfg.MODEL.POINTREND.BASEMODEL
        #self.backbone使用的是Xception65的Deeplabv3+
        self.backbone =  MODEL_REGISTRY.get(model_name)()
        #self.head是论文中主要的创新
        self.head = PointHead(num_classes=self.nclass)

    def forward(self, x):   # x:[2, 3, 768, 768], 传入的图片
        c1, _, _, c4 = self.backbone.encoder(x)   #backbone.encoder使用的是Xception网络
        #c1:[2, 256, 192, 192] c4:[2, 2048, 48, 48] 是Xception不同网络层的特征

        out = self.backbone.head(c4, c1)    #out：[2, 19, 48, 48],粗糙分类的结果
        
        result = {'res2': c1, 'coarse': out}
        #result['res2']：[2, 256, 192, 192],表示xception的第一层特征输出
        #result['coarse']:[2, 19, 48, 48]表示经过级联空洞卷积提取的特征的粗糙预测

        result.update(self.head(x, result["res2"], result["coarse"])) #self.head函数是论文核心

        if not self.training:
            return (result['fine'],)
        return result
    #result{'res2': c1 feats[1, 256, 192, 192], 'coarse': coarse classifer[2, 19, 48, 48],
    # 'rend':uncertain point classifer[2, 19, 48], 'points': uncertain point position[2, 48, 2]}


class PointHead(nn.Module):
    """
    主要思路：相比较于deeplabv3+方法拿到图片特征直接插值的方法。这里通过sampling_points函数，
    找到最不稳定的像素点，使用mlp进行进一步判断，来增强这些不稳定像素点的准确度。
    """
    def __init__(self, in_c=275, num_classes=19, k=3, beta=0.75):
        super().__init__()
        self.mlp = nn.Conv1d(in_c, num_classes, 1)
        self.k = k
        self.beta = beta

    def forward(self, x, res2, out):
        """
        主要思路：
        通过 out（粗糙预测）计算出top N 个不稳定的像素点，针对每个不稳定像素点得到在res2（fine）
        和out（coarse）中对应的特征，组合N个不稳定像素点对应的fine和coarse得到rend，
        再通过mlp得到更准确的预测
        :param x: 表示输入图片的特征     eg.[2, 3, 768, 768]
        :param res2: 表示xception的第一层特征输出     eg.[2, 256, 192, 192]
        :param out: 表示经过级联空洞卷积提取的特征的粗糙预测    eg.[2, 19, 48, 48]
        :return: rend:更准确的预测，points：不确定像素点的位置
        """
        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """
        if not self.training:
            return self.inference(x, res2, out)

        points = sampling_points(out, x.shape[-1] // 16, self.k, self.beta) #out:[2, 19, 48, 48] || x:[2, 3, 768, 768] || points:[2, 48, 2]

        coarse = point_sample(out, points, align_corners=False) #[2, 19, 48]
        fine = point_sample(res2, points, align_corners=False)  #[2, 256, 48]

        feature_representation = torch.cat([coarse, fine], dim=1)   #[2, 275, 48]

        rend = self.mlp(feature_representation) #[2, 19, 48]

        return {"rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, x, res2, out):
        """
        输入：
        x:[1, 3, 768, 768],表示输入图片的特征
        res2:[1, 256, 192, 192]，表示xception的第一层特征输出
        out:[1, 19, 48, 48],表示经过级联空洞卷积提取的特征的粗糙预测
        输出：
        out:[1,19,768,768],表示最终图片的预测
        主要思路：
        通过 out计算出top N = 8096 个不稳定的像素点，针对每个不稳定像素点得到在res2（fine）
        和out（coarse）中对应的特征，组合8096个不稳定像素点对应的fine和coarse得到rend，
        再通过mlp得到更准确的预测，迭代至rend的尺寸大小等于输入图片的尺寸大小
        """
        """
        During inference, subdivision uses N=8096
        (i.e., the number of points in the stride 16 map of a 1024×2048 image)
        """
        num_points = 8096
        
        while out.shape[-1] != x.shape[-1]: #out:[1, 19, 48, 48], x:[1, 3, 768, 768]
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)   #out[1, 19, 48, 48]

            points_idx, points = sampling_points(out, num_points, training=self.training)   #points_idx:8096 || points:[1, 8096, 2]

            coarse = point_sample(out, points, align_corners=False) #coarse:[1, 19, 8096]   表示8096个不稳定像素点根据高级特征得出的对应的类别
            fine = point_sample(res2, points, align_corners=False)  #fine:[1, 256, 8096]    表示8096个不稳定像素点根据低级特征得出的对应类别

            feature_representation = torch.cat([coarse, fine], dim=1)   #[1, 275, 8096] 表示8096个不稳定像素点合并fine和coarse的特征

            rend = self.mlp(feature_representation) #[1, 19, 8096]

            B, C, H, W = out.shape  #first:[1, 19, 128, 256]
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)  #[1, 19, 8096]
            out = (out.reshape(B, C, -1)
                      .scatter_(2, points_idx, rend)    #[1, 19, 32768]
                      .view(B, C, H, W))    #[1, 19, 128, 256]
            
        return {"fine": out}


def point_sample(input, point_coords, **kwargs):
    """
    主要思路：通过不确定像素点的位置信息，得到不确定像素点在input特征层上的对应特征
    :param input: 图片提取的特征（res2、out） eg.[2, 19, 48, 48]
    :param point_coords: 不确定像素点的位置信息 eg.[2, 48, 2], 2:batch_size, 48:不确定点的数量，2:空间相对坐标
    :return: 不确定像素点在input特征层上的对应特征 eg.[2, 19, 48]
    """
    """
    From Detectron2, point_features.py#19
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3: #point_coords:[2, 48, 2]
        add_dim = True
        point_coords = point_coords.unsqueeze(2)    #point_coords:[2, 48, 1, 2]
    output = F.grid_sample(input, 2.0 * point_coords - 1.0)#, **kwargs) #output:[2, 19, 48, 1] || input:[2, 19, 48, 48]
    if add_dim:
        output = output.squeeze(3)  #output:[2, 19, 48]
    return output


@torch.no_grad()
def sampling_points(mask, N, k=3, beta=0.75, training=True):
    """
    主要思想：根据粗糙的预测结果，找出不确定的像素点
    :param mask: 粗糙的预测结果（out）   eg.[2, 19, 48, 48]
    :param N: 不确定点个数（train：N = 图片的尺寸/16, test: N = 8096）    eg. N=48
    :param k: 论文超参
    :param beta: 论文超参
    :param training:
    :return: 不确定点的位置坐标  eg.[2, 48, 2]
    """
    """
    Follows 3.1. Point Selection for Inference and Training
    In Train:, `The sampling strategy selects N points on a feature map to train on.`
    In Inference, `then selects the N most uncertain points`
    Args:
        mask(Tensor): [B, C, H, W]
        N(int): `During training we sample as many points as there are on a stride 16 feature map of the input`
        k(int): Over generation multiplier
        beta(float): ratio of importance points
        training(bool): flag
    Return:
        selected_point(Tensor) : flattened indexing points [B, num_points, 2]
    """
    assert mask.dim() == 4, "Dim must be N(Batch)CHW"   #this mask is out(coarse)
    device = mask.device
    B, _, H, W = mask.shape   #first: mask[1, 19, 48, 48]
    mask, _ = mask.sort(1, descending=True) #_ : [1, 19, 48, 48],按照每一类的总体得分排序

    if not training:
        H_step, W_step = 1 / H, 1 / W
        N = min(H * W, N)
        uncertainty_map = -1 * (mask[:, 0] - mask[:, 1])
        #mask[:, 0]表示每个像素最有可能的分类，mask[:, 1]表示每个像素次有可能的分类，当一个像素
        #即是最有可能的又是次有可能的，则证明它不好预测，对应的uncertainty_map就相对较大
        _, idx = uncertainty_map.view(B, -1).topk(N, dim=1) #id选出最不好预测的N个点

        points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        points[:, :, 0] = W_step / 2.0 + (idx  % W).to(torch.float) * W_step    #点的横坐标
        points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step    #点的纵坐标
        return idx, points  #idx:48 || points:[1, 48, 2]

    # Official Comment : point_features.py#92
    # It is crucial to calculate uncertanty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to worse results. To illustrate the difference: a sampled point between two coarse predictions
    # with -1 and 1 logits has 0 logit prediction and therefore 0 uncertainty value, however, if one
    # calculates uncertainties for the coarse predictions first (-1 and -1) and sampe it for the
    # center point, they will get -1 unceratinty.

    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners=False)

    uncertainty_map = -1 * (over_generation_map[:, 0] - over_generation_map[:, 1])
    _, idx = uncertainty_map.topk(int(beta * N), -1)

    shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)

    idx += shift[:, None]

    importance = over_generation.view(-1, 2)[idx.view(-1), :].view(B, int(beta * N), 2)
    coverage = torch.rand(B, N - int(beta * N), 2, device=device)
    return torch.cat([importance, coverage], 1).to(device)
