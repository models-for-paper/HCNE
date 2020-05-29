import torch
import torch.nn as nn
import math


class CHE(nn.Module):
    def __init__(self, node_num, hidden_dim, radius_scale_factor, radius_0, pad_id):
        super(CHE, self).__init__()
        self.radius_scale_factor = radius_scale_factor  # alpha
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.init_radius = radius_0
        # self.init_radius_weigth = torch.ones(node_num, hidden_dim)*radius_0
        self.radius_emb = nn.Embedding(node_num, hidden_dim, padding_idx=pad_id)
        self.angle_emb = nn.Embedding(node_num, hidden_dim, padding_idx=pad_id)

        self.complex_coordinates_real = nn.Embedding(node_num, hidden_dim, padding_idx=pad_id)
        self.complex_coordinates_img = nn.Embedding(node_num, hidden_dim, padding_idx=pad_id)


    def radius_scale_function(self, parents_radii, angles):
        """generate the children radii by the parents radii
        Arguments:
            radii {[type]} -- [description]
            angles {[type]} -- [description]
        """
        return self.radius_scale_factor * 0.5 * parents_radii * torch.abs(torch.sin(0.5 * angles))

    def forward(self, children, brothers, parents, brothers_parents, unbrothers):
        parents_radii = torch.relu(self.radius_emb(parents))
        children_radii_true = torch.relu(self.radius_emb(children))  # \rho

        parents_angles = self.angle_emb(parents) % (2*math.pi)
        brothers_parents_angles = self.angle_emb(brothers_parents) % (2*math.pi)
        diff_angle = parents_angles - brothers_parents_angles

        # ralative coordinates
        children_angles = self.angle_emb(children) % (2*math.pi)
        children_radii = self.radius_scale_function(parents_radii, diff_angle)  # f(\rho)

        # parrents coordinates
        real_p = self.complex_coordinates_real(parents)
        img_p =  self.complex_coordinates_img(parents)

        # calculated children coordinates
        _real_, _img_ = children_radii * torch.cos(children_angles), children_radii * torch.sin(children_angles)
        _real_c, _img_c = _real_ + real_p, _img_ + img_p
        # learned coordinates
        real_c = self.complex_coordinates_real(children)
        img_c =  self.complex_coordinates_img(children)
        # brother sample
        brothers_real = self.complex_coordinates_real(brothers)
        brothers_img = self.complex_coordinates_img(brothers)
        # negative sample
        unbrothers_real = self.complex_coordinates_real(unbrothers)
        unbrothers_img = self.complex_coordinates_img(unbrothers)

        return _real_c, _img_c, real_c, img_c, children_radii, children_radii_true, unbrothers_real, unbrothers_img, brothers_real, brothers_img

    # def forward(self, children, parents):
    #     parents_radii = self.radius_emb(parents)
    #     parents_angles = self.angle_emb(parents) % (2*math.pi)
    #     children_radii = self.radius_scale_function(parents_radii, parents_angles)




        

