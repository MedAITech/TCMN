import torch
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
import torch.nn as nn
import torch.nn.functional as F
from util.tensor_util import compute_tensor_iu

from collections import defaultdict


def get_iou_hook(values):
    return 'iou/iou', (values['hide_iou/i']+1)/(values['hide_iou/u']+1)

def get_sec_iou_hook(values):
    return 'iou/sec_iou', (values['hide_iou/sec_i']+1)/(values['hide_iou/sec_u']+1)

iou_hooks_so = [
    get_iou_hook,
]

iou_hooks_mo = [
    get_iou_hook,
    get_sec_iou_hook,
]

def similar_loss(v1,v2,v3):
    d12 = torch.pow(v1 - v2, 2)
    d23 = torch.pow(v2 - v3, 2)
    d13 = torch.pow(v1 - v3, 2)
    exp_sum = torch.mean(d12)+torch.mean(d23)
    exp_13 = torch.mean(d13)
    return exp_sum/exp_13


def similar_loss2(v1,v2,v3):
    b = v1.shape[0]
    v1 = v1.view(b,-1)
    v2 = v2.view(b,-1)
    v3 = v3.view(b,-1)
    d12 = 1 - F.cosine_similarity(v1,v2,dim=1)
    d23 = 1 - F.cosine_similarity(v2,v3,dim=1)
    d13 = 1 - F.cosine_similarity(v1,v3,dim=1)
    exp_sum = torch.mean(d12)+torch.mean(d23)
    exp_13 = torch.mean(d13)
    return exp_sum/exp_13

def jenson_shannon_divergence(net_1_probs, net_2_probs):
    total_m = 0.5 * (net_1_probs + net_2_probs)
    loss = F.kl_div(net_1_probs.log(), total_m) + F.kl_div(net_2_probs.log(), total_m)
    return 0.5 * loss

class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]  # 这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3

    def simplex(self, t: Tensor, axis=1) -> bool:
        _sum = t.sum(axis).type(torch.float32)
        _ones = torch.ones_like(_sum, dtype=torch.float32)
        return torch.allclose(_sum, _ones)

    def sset(self, a: Tensor, sub: Iterable) -> bool:
        return self.uniq(a).issubset(sub)

    def uniq(self, a: Tensor) -> Set:
        return set(torch.unique(a.cpu()).numpy())

    def one_hot(self, t: Tensor, axis=1) -> bool:
        return self.simplex(t, axis) and self.sset(t, [0, 1])

    def class2one_hot(self, seg: Tensor, C: int) -> Tensor:
        if len(seg.shape) == 2:  # Only w, h, used by the dataloader
            seg = seg.unsqueeze(dim=0)
        assert self.sset(seg, list(range(C)))

        b, w, h = seg.shape  # type: Tuple[int, int, int]

        res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
        assert res.shape == (b, C, w, h)
        assert self.one_hot(res)

        return res

    def one_hot2dist(self, seg: np.ndarray) -> np.ndarray:
        assert self.one_hot(torch.Tensor(seg), axis=0)
        C: int = len(seg)

        res = np.zeros_like(seg)
        # res = res.astype(np.float64)
        for c in range(C):
            posmask = seg[c].astype(bool)

            if posmask.any():
                negmask = ~posmask
                res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
        return res

    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, gt: Tensor, _: Tensor) -> Tensor:
        gt_one_hot = self.class2one_hot(gt,2)  # b,n,h,w
        gt_one_hot = gt_one_hot[0].cpu().numpy()  #n,h,w
        dist_maps = self.one_hot2dist(gt_one_hot)
        dist_maps = torch.tensor(dist_maps).unsqueeze(0)
        probs = F.softmax(probs, dim=1)
        assert self.simplex(probs)
        assert not self.one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32).cuda()
        dc = dist_maps[:, self.idc, ...].type(torch.float32).cuda()

        multiplied = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multiplied.mean()

        return loss


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class LossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        if not self.para['topp']:
            print('--------------------------Top P Deny---------------------------')
            self.bce = BootstrappedCE(top_p=1.0)
        else:
            self.bce = BootstrappedCE(start_warm=para['iterations']*para['topp_start_warm'],end_warm=para['iterations']*para['topp_end_warm'],top_p=para['topp_end_ratio'])
        self.coarse_ce = BootstrappedCE(top_p=1.0)
        self.boundary = SurfaceLoss()
    def compute(self, data, it):
        losses = defaultdict(int)

        b, s, _, _, _ = data['gt'].shape
        selector = data.get('selector', None)
        precision = []
        recall = []

        # 训练的时候是随机选取第一帧中的两个物体进行训练（但是实际上是一个个） 但valueEncoder有问题（会融合两个mask） 训练和测试需要改
        # 测试时是一个一个物体分割 valueEncoder会以某种方法将不是该物体的别的mask融合在一起
        for i in range(1, s):  # 第一帧是给出的，所以从第二帧开始算损失
            # Have to do it in a for-loop like this since not every entry has the second object
            # Well it's not a lot of iterations anyway
            for j in range(b):
                if selector is not None and selector[j][1] > 0.5:  # 有第二个物体，取全部做损失
                    loss, p = self.bce(data['logits_%d'%i][j:j+1], data['cls_gt'][j:j+1,i], it)
                    loss_coarse, p2 = self.bce(data['logits_coarse_%d'%i][j:j+1], data['cls_gt'][j:j+1,i], it)
                else:                                              # 无第二个物体，取部分做损失，这样不会因为第二个对象为空模型受到影响
                    loss, p = self.bce(data['logits_%d'%i][j:j+1,:2], data['cls_gt'][j:j+1,i], it)
                    # loss_coarse, p2 = self.bce(data['logits_coarse_%d'%i][j:j+1,:2], data['cls_gt'][j:j+1,i], it)
                    # loss_boundary = self.boundary(data['logits_%d'%i][j:j+1,:2],data['cls_gt'][j:j+1,i],None)
                    # loss_coarse_boundary = self.boundary(data['logits_coarse_%d'%i][j:j+1,:2],data['cls_gt'][j:j+1,i],None)
                    # loss_contrastive, p3 = self.bce(data['logits_coarse_%d'%i][j:j+1,:2], torch.concat((1-data['mask_%d'%i][j:j+1,:2],data['mask_%d'%i][j:j+1,:2]),dim=1), it)
                    # loss_contrastive = jenson_shannon_divergence(data['mask_coarse_%d'%i][j:j+1], data['mask_%d'%i][j:j+1])

                losses['loss_%d'%i] += loss / b
                # losses['loss_coarse_%d'%i] += loss_coarse / b
                # losses['loss_boundary_%d'%i] += loss_boundary / b
                # losses['loss_coarse_boundary_%d'%i] += loss_coarse_boundary / b

                # losses['loss_contrastive_%d'%i] += loss_contrastive / b
                losses['p'] += p / b / (s-1)

            losses['total_loss'] += losses['loss_%d'%i]
            # losses['total_loss'] += 0.1 * losses['loss_coarse_%d'%i]
            # losses['total_loss'] += 0.1 * losses['loss_boundary_%d'%i]
            # losses['total_loss'] += 0.1 * losses['loss_coarse_boundary_%d'%i]
            # losses['total_loss'] += 0.1 * losses['loss_contrastive_%d'%i]

            new_total_i, new_total_u = compute_tensor_iu(data['mask_%d'%i]>0.5, data['gt'][:,i]>0.5)
            losses['hide_iou/i'] += new_total_i
            losses['hide_iou/u'] += new_total_u
            precision.append(new_total_i/(data['mask_%d'%i]>0.5).int().sum())
            recall.append(new_total_i/(data['gt'][:,i]>0.5).int().sum())
            if selector is not None and not self.para['single_object'] > 0.5:
                new_total_i, new_total_u = compute_tensor_iu(data['sec_mask_%d'%i]>0.5, data['sec_gt'][:,i]>0.5)
                losses['hide_iou/sec_i'] += new_total_i
                losses['hide_iou/sec_u'] += new_total_u
        v1 = data['v1']
        v2 = data['v2']
        v3 = data['v3']
        # losses['loss_similar'] = 0.1 * similar_loss2(v1,v2,v3)
        losses['total_loss'] += losses['loss_similar']
        losses['precision'] = sum(precision)/len(precision)
        losses['recall'] = sum(recall)/len(recall)
        losses['f1_score'] = 2*(losses['precision']*losses['recall'])/(losses['precision']+losses['recall'])
        losses['dice'] = 2*(losses['hide_iou/i'])/(losses['hide_iou/i']+losses['hide_iou/u'])
        losses['iou'] = losses['hide_iou/i']/losses['hide_iou/u']

        return losses
