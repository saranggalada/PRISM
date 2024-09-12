### Data prep

import os
import numpy as np
import re
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Pad, CenterCrop, ToTensor, ToPILImage, Resize
import nibabel as nib
from glob import glob
from augmentations import *

default_transform = Compose([ToPILImage(), Pad(63), CenterCrop([256, 256])])
contrast_names = ['T2', 'PD']#, 'T1', 'FLAIR']

def get_mri_slice(fpath):
    if os.path.exists(fpath):
        image = np.squeeze(nib.load(fpath).get_fdata().astype(np.float32))#.transpose([1, 0, 2])
        # print(image.shape)

        image = np.rot90(image[:, :, image.shape[2] // 2])
        image = image / np.max(image)
        image = np.array(default_transform(image))
        image = ToTensor()(image)

    else:
        print("EMPTY IMAGE")
        image = torch.ones([1, 256, 256])
        
    return image


def remove_background(image_dicts):
    num_contrasts = len(contrast_names)
    cutoffs = [5e-2, 5e-2, 5e-2]
    masks = [torch.ones((1, 256, 256)), torch.ones((1, 256, 256)), torch.ones((1, 256, 256))]
    # for contrast_id, image_dict in enumerate(image_dicts):
    for i in range(num_contrasts):
        # masks[image_dict['contrast_id']] = masks[image_dict['contrast_id']] * image_dict['image'].ge(cutoffs[image_dict['contrast_id']])
        masks[i] = masks[i] * image_dicts[i]['image'].ge(cutoffs[i])
        image_dicts[i]['image'] = image_dicts[i]['image'] * masks[i]
        image_dicts[i]['mask'] = masks[i].bool()
        for j in range(len(image_dicts[i]['aug'])):
            image_dicts[i]['aug'][j] = image_dicts[i]['aug'][j] * masks[i]
    return image_dicts


class PRISM_MRI_Dataset(Dataset):
    def __init__(self, dataset_dir, mode='train'):#, orientations=['axial', 'coronal', 'sagittal']):
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.t2_paths = self._get_file_paths()

        self.dataset = []

        for idx in range(len(self.t2_paths)):
            image_dicts = []
            filename = self.t2_paths[idx][-26:]
            subject_id = re.search(r'IXI(.*?)-', filename).group(1).strip()
            site_name = re.search(r'-(.*?)-', filename).group(1).strip()
            for contrast_name in contrast_names:
                image_path = self.t2_paths[idx].replace('T2', contrast_name)
                image = get_mri_slice(image_path)
                img = image.squeeze().numpy() * 255
                augs = [gamma_correction(img, 0.5), gamma_correction(img, 1.5), apply_bias_fields(img, 3, 0.5, contrast_name), add_gaussian_noise(img, 0, 5)]
                augs = [ToTensor()(aug) for aug in augs]
                image_dict = {'image': image,
                            'aug': augs,
                            'site_name': site_name,
                            'subject_id': subject_id,
                            'modality': contrast_name,
                            'exists': 0 if image[0, 0, 0] > 0.9999 else 1}
                image_dicts.append(image_dict)
            self.dataset.append(remove_background(image_dicts))

    def _get_file_paths(self):
        fpaths = []
        fpaths = sorted(glob(os.path.join(self.dataset_dir, self.mode, f'*T2*nii.gz')))
        # print(f'dataset dir: {dataset_dir}')
        return fpaths

    def __len__(self):
        return len(self.t2_paths)

    def __getitem__(self, idx: int):
        return self.dataset[idx]