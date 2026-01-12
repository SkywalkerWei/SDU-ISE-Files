import torch
import torch.nn as nn

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    box2 = box2.T
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (4 / torch.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
        else:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
    return iou

class ComputeLoss:
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device
        
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        
        self.BCEcls, self.BCEobj = BCEcls, BCEobj
        self.gr = 1.0
        self.autobalance = autobalance
        self.na = model.detect.na
        self.nc = model.detect.nc
        self.nl = model.detect.nl
        self.anchors = model.detect.anchors
        self.device = device
        
    def __call__(self, p, targets):
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        
        tcls, tbox, indices, anchors = self.build_targets(p, targets)
        
        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=self.device)

            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]
                
                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                # Classification
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], 0, device=self.device)
                    t[range(n), tcls[i]] = 1.0
                    lcls += self.BCEcls(ps[:, 5:], t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji
        
        lbox *= 0.05
        lobj *= 1.0
        lcls *= 0.5
        
        return (lbox + lobj + lcls) * targets.shape[0], torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        
        g = 0.5
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * g

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
            
            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < 4.0
                t = t[j]
                
                gxy = t[:, 2:4]
                gxi = gain[2:4] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = gxy - offsets
            
            # Integer part of grid indices for tbox calculation
            gij_long = gij.long()
            gi, gj = gij.T
            
            # Anchor index
            a = t[:, 6].long()

            # Clamp float grid indices, then convert to long. This is the robust way.
            gi = gi.clamp(0, gain[2] - 1).long()
            gj = gj.clamp(0, gain[3] - 1).long()
            
            # Append final long indices
            indices.append((b, a, gj, gi))
            
            # Append regression targets
            tbox.append(torch.cat((gxy - gij_long, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch