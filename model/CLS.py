import torch
import torch.nn as nn
import math


class CLS(nn.Module):
    def __init__(self, model, hidden_dim, class_num, coor_dim=2):
        super(CLS, self).__init__()
        self.model = model
        self.class_num = class_num
        self.coor_dim = coor_dim
        self.hidden_dim = hidden_dim

        self.classfication_function = nn.Linear(coor_dim*hidden_dim, class_num)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, children, brothers, parents, brothers_parents, unbrothers, need_loss=True):
        _real_c, _img_c, real_c, img_c, children_radii, children_radii_true, \
            unbrothers_real, unbrothers_img, brothers_real, brothers_img = \
            self.model(children, brothers, parents, brothers_parents, unbrothers)
        _feature = torch.cat((_real_c, _img_c), dim=-1)
        feature = torch.cat((real_c, img_c), dim=-1)
        brothers_feature = torch.cat((brothers_real, brothers_img),dim=-1)
        unbrothers_feature = torch.cat((unbrothers_real, unbrothers_img),dim=-1)
        logits = self.classfication_function(feature)
        esp = 1e-9
        if need_loss:
            # local similarity
            loss_local = torch.mean(torch.norm(children_radii-children_radii_true, 2, -1) + \
                torch.norm(feature - _feature + esp, 2,-1))
            # global structure strength
            loss_global = torch.mean((2.0)/(2.0-self.model.radius_scale_factor)*self.model.init_radius -\
                torch.norm(feature, 2, dim=-1))
            # first order
            loss_1 = -1.0*torch.log(torch.sigmoid(torch.bmm(feature.unsqueeze(1), brothers_feature.unsqueeze(-1)).squeeze())).mean()
            # second order
            loss_2 = -1.0*(torch.log(torch.sigmoid(torch.bmm(feature.unsqueeze(1), brothers_feature.unsqueeze(-1)).squeeze())).mean() +\
                torch.log(torch.sigmoid(-1.0 * torch.bmm(unbrothers_feature, feature.unsqueeze(-1))).squeeze(-1)).sum(-1).mean())

            # print(torch.log(torch.sigmoid(-1.0 * torch.bmm(unbrothers_feature, feature.unsqueeze(-1))).squeeze(-1)))
            return logits, loss_1+loss_local+0.01*loss_global, feature

            # return logits, loss_2, feature
        else:
            return logits, feature
