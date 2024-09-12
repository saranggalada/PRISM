### Harmonize.py

import torch
from model import PRISM
from dataprep import PRISM_MRI_Dataset

# load datasets pretrained models (if training completed)
out_dir = '/kaggle/working/'

guys_test_ds = torch.load('/kaggle/input/ixi-guys-train/ixi-guys-ds.pth')
hh_test_ds = torch.load('/kaggle/input/ixi-hh-train/ixi-hh-train.pth')
iop_test_ds = torch.load('/kaggle/input/ixi-iop-train/ixi-iop-train.pth')

# Choose Target site as Guys
# All other sites will now use Guys' style encoder and decoder

# Load the trained PRISM model for site Guys
prism_g = PRISM(intensity_levels=5, latent_dim=2, num_sites=3, gpu_id=0, modality='T2', modalities = ['T2', 'PD'])
prism_g.anatomy_encoder.load_state_dict(torch.load(f'{out_dir}/prism_anatomy_encoder_guys.pth'))
prism_g.style_encoder.load_state_dict(torch.load(f'{out_dir}/prism_style_encoder_guys.pth'))
prism_g.decoder.load_state_dict(torch.load(f'{out_dir}/prism_decoder_guys.pth'))

# Load the trained PRISM model for site HH
prism_h = PRISM(intensity_levels=5, latent_dim=2, num_sites=3, gpu_id=0, modality='T2', modalities = ['T2', 'PD'])
prism_h.anatomy_encoder.load_state_dict(torch.load(f'{out_dir}/prism_anatomy_encoder_hh.pth'))

# Load the trained PRISM model for site IOP
prism_i = PRISM(intensity_levels=5, latent_dim=2, num_sites=3, gpu_id=0, modality='T2', modalities = ['T2', 'PD'])
prism_i.anatomy_encoder.load_state_dict(torch.load(f'{out_dir}/prism_anatomy_encoder_iop.pth'))


style_codes_guys = []
style_codes_hh = []
style_codes_iop = []

anatomies_guys = []
anatomies_hh = []
anatomies_iop = []

mask_guys = []
mask_hh = []
mask_iop = []

names_guys = []
names_hh = []
names_iop = []

with torch.set_grad_enabled(False):
    prism_g.anatomy_encoder.eval()
    prism_h.anatomy_encoder.eval()
    prism_i.anatomy_encoder.eval()
    prism_g.style_encoder.eval()

    # Guys
    for subject in guys_test_ds:
        image = subject[prism_g.modality]['image'].to(prism_g.device).unsqueeze(1)
        mask = subject[prism_g.modality]['mask'].to(prism_g.device).unsqueeze(1)
        anatomy = prism_g.get_anatomy_representations(image, mask)
        style, _, _ = prism_g.get_style_code(image)
        anatomies_guys.append(anatomy.detach().squeeze())
        style_codes_guys.append(style.detach().cpu().squeeze())
        mask_guys.append(mask.detach().squeeze())
        names_guys.append(subject[prism_g.modality]['subject_id'])

    # HH
    for subject in hh_test_ds:
        image = subject[prism_h.modality]['image'].to(prism_h.device).unsqueeze(1)
        mask = subject[prism_h.modality]['mask'].to(prism_h.device).unsqueeze(1)
        anatomy = prism_h.get_anatomy_representations(image, mask)
        style, _, _ = prism_g.get_style_code(image) # Use Guys' style encoder
        anatomies_hh.append(anatomy.detach().squeeze())
        style_codes_hh.append(style.detach().cpu().squeeze())
        mask_hh.append(mask.detach().squeeze())
        names_hh.append(subject[prism_h.modality]['subject_id'])

    # IOP
    for subject in iop_test_ds:
        image = subject[prism_i.modality]['image'].to(prism_i.device).unsqueeze(1)
        mask = subject[prism_i.modality]['mask'].to(prism_i.device).unsqueeze(1)
        anatomy = prism_i.get_anatomy_representations(image, mask)
        style, _, _ = prism_g.get_style_code(image) # Use Guys' style encoder
        anatomies_iop.append(anatomy.detach().squeeze())
        style_codes_iop.append(style.detach().cpu().squeeze())
        mask_iop.append(mask.detach().squeeze())
        names_iop.append(subject[prism_i.modality]['subject_id'])

import matplotlib.pyplot as plt

style_codes_guys = torch.stack(style_codes_guys)
style_codes_hh = torch.stack(style_codes_hh)
style_codes_iop = torch.stack(style_codes_iop)


# TSNE Plot to visualise harmonization
plt.figure(figsize=(12, 6))
plt.scatter(style_codes_guys[:, 0], style_codes_guys[:, 1], c='blue', label='Guys')
plt.scatter(style_codes_hh[:, 0], style_codes_hh[:, 1], c='red', label='HH')
plt.scatter(style_codes_iop[:, 0], style_codes_iop[:, 1], c='green', label='IOP')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('Style codes site-wise')
plt.legend()
plt.show()


style_code_guys = torch.mean(style_codes_guys, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(prism_g.device)
style_code_hh = torch.mean(style_codes_hh, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(prism_h.device)
style_code_iop = torch.mean(style_codes_iop, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(prism_i.device)

rec_guys = []
rec_hh = []
rec_iop = []


with torch.set_grad_enabled(False):
    prism_g.decoder.eval()
    
    # Guys
    for i in range(len(anatomies_guys)):
        anatomy = anatomies_guys[i].unsqueeze(0).unsqueeze(0)
        mask = mask_guys[i].unsqueeze(0)
        rec_image = prism_g.decode(anatomy, style_code_guys, mask)
        rec_guys.append(rec_image.detach())
        plt.imsave(f'{out_dir}/T2/Guys/{names_guys[i]}_harmonized_to_Guys.png', rec_image.squeeze().cpu().numpy(), cmap='gray')

    # HH
    for i in range(len(anatomies_hh)):
        anatomy = anatomies_hh[i].unsqueeze(0).unsqueeze(0)
        mask = mask_hh[i].unsqueeze(0)
        rec_image = prism_g.decode(anatomy, style_code_hh, mask)
        rec_hh.append(rec_image.detach())
        plt.imsave(f'{out_dir}/T2/HH/{names_hh[i]}_harmonized_to_Guys.png', rec_image.squeeze().cpu().numpy(), cmap='gray')

    # IOP
    for i in range(len(anatomies_iop)):
        anatomy = anatomies_iop[i].unsqueeze(0).unsqueeze(0)
        mask = mask_iop[i].unsqueeze(0)
        rec_image = prism_g.decode(anatomy, style_code_iop, mask)
        rec_iop.append(rec_image.detach())
        plt.imsave(f'{out_dir}/T2/IOP/{names_iop[i]}_harmonized_to_Guys.png', rec_image.squeeze().cpu().numpy(), cmap='gray')