### Losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg = nn.Sequential(*list(vgg_model.children())[:13]).eval()

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        return F.l1_loss(self.vgg(x), self.vgg(y))


class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature

    def forward(self, query_feature, positive_feature, negative_feature):
        B, C, N_query = query_feature.shape
        _, _, N_positive = positive_feature.shape
        _, _, N_negative = negative_feature.shape

        # Reshape query_feature to [B*N_query, C, 1]
        query_feature = query_feature.permute(0, 2, 1).reshape(B*N_query, C, 1)

        # Reshape positive_feature to [B*N_query, C, N_positive]
        positive_feature = positive_feature.unsqueeze(1).expand(-1, N_query, -1, -1)
        positive_feature = positive_feature.reshape(B*N_query, C, N_positive)

        # Compute l_positive
        l_positive = torch.bmm(query_feature.transpose(1, 2), positive_feature).squeeze(1)

        # Reshape negative_feature to [B*N_query, C, N_negative]
        negative_feature = negative_feature.unsqueeze(1).expand(-1, N_query, -1, -1)
        negative_feature = negative_feature.reshape(B*N_query, C, N_negative)

        # Compute l_negative
        l_negative = torch.bmm(query_feature.transpose(1, 2), negative_feature).squeeze(1)

        # Concatenate l_positive and l_negative
        logits = torch.cat((l_positive, l_negative), dim=1) / self.temperature

        # Create targets
        targets = torch.zeros(B * N_query, dtype=torch.long, device=query_feature.device)

        return self.ce_loss(logits, targets).mean()


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        kld_loss = -0.5 * logvar + 0.5 * (torch.exp(logvar) + torch.pow(mu, 2)) - 0.5
        return kld_loss


def divide_into_batches(in_tensor, num_batches):
    batch_size = in_tensor.shape[0] // num_batches
    remainder = in_tensor.shape[0] % num_batches
    batches = []

    current_start = 0
    for i in range(num_batches):
        current_end = current_start + batch_size
        if remainder:
            current_end += 1
            remainder -= 1
        batches.append(in_tensor[current_start:current_end, ...])
        current_start = current_end
    return batches


def normalize_intensity(image):
    thresh = np.percentile(image.flatten(), 95)
    image = image / (thresh + 1e-5)
    image = np.clip(image, a_min=0.0, a_max=5.0)
    return image, thresh


def zero_pad(image, image_dim=256):
    [n_row, n_col, n_slc] = image.shape
    image_padded = np.zeros((image_dim, image_dim, image_dim))
    center_loc = image_dim // 2
    image_padded[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2,
                 center_loc - n_slc // 2: center_loc + n_slc - n_slc // 2] = image
    return image_padded


def crop(image, n_row, n_col, n_slc):
    image_dim = image.shape[0]
    center_loc = image_dim // 2
    return image[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2,
                 center_loc - n_slc // 2: center_loc + n_slc - n_slc // 2]
