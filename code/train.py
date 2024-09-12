### Train.py

import torch
from model import PRISM

torch.autograd.set_detect_anomaly(True)

out_dir = '/kaggle/working/'
train_path_g = '/kaggle/input/ixi-guys-train/ixi-guys-train.pth'
test_path_g = '/kaggle/input/ixi-guys-test/ixi-guys-test.pth'
train_path_h = '/kaggle/input/ixi-hh-train/ixi-hh-train.pth'
test_path_h = '/kaggle/input/ixi-hh-test/ixi-hh-test.pth'
train_path_i = '/kaggle/input/ixi-iop-train/ixi-iop-train.pth'
test_path_i = '/kaggle/input/ixi-iop-test/ixi-iop-test.pth'

lr = 5e-4
batch_size = 8
epochs = 50
gpu_id = 0

print(f'{'=' * 10} PRISM training starts {'=' * 10}')

# ====== 1. INITIALIZE MODEL ======
prism_g = PRISM(intensity_levels=5, latent_dim=2, num_sites=3, gpu_id=gpu_id, modality='T2', modalities = ['T2', 'PD'])
prism_h = PRISM(intensity_levels=5, latent_dim=2, num_sites=3, gpu_id=gpu_id, modality='T2', modalities = ['T2', 'T1'])
prism_i = PRISM(intensity_levels=5, latent_dim=2, num_sites=3, gpu_id=gpu_id, modality='T2', modalities = ['T2', 'T1', 'PD'])

# ====== 2. LOAD DATASETS ======
prism_g.load_dataset_from_pt(batch_size, train_path=train_path_g, test_path=test_path_g)
prism_h.load_dataset_from_pt(batch_size, train_path=train_path_h, test_path=test_path_h)
prism_i.load_dataset_from_pt(batch_size, train_path=train_path_i, test_path=test_path_i)

# ====== 3. INITIALIZE TRAINING ======
prism_g.init_training(out_dir=out_dir, lr=lr, vgg_path='/kaggle/input/vgg16-imagenet/pytorch/default/1/vgg16_imagenet.pth')
prism_h.init_training(out_dir=out_dir, lr=lr, vgg_path='/kaggle/input/vgg16-imagenet/pytorch/default/1/vgg16_imagenet.pth')
prism_i.init_training(out_dir=out_dir, lr=lr, vgg_path='/kaggle/input/vgg16-imagenet/pytorch/default/1/vgg16_imagenet.pth')

# ====== 4. BEGIN TRAINING ======
prism_g.train(epochs=epochs)
prism_h.train(epochs=epochs)
prism_i.train(epochs=epochs)

# ====== 5. SAVE MODELS ======
# Site Guys
torch.save(prism_g.anatomy_encoder.state_dict(), f'{prism_g.out_dir}/prism_anatomy_encoder_guys.pth')
torch.save(prism_g.style_encoder.state_dict(), f'{prism_g.out_dir}/prism_style_encoder_guys.pth')
torch.save(prism_g.decoder.state_dict(), f'{prism_g.out_dir}/prism_decoder_guys.pth')

# Site HH
torch.save(prism_h.anatomy_encoder.state_dict(), f'{prism_h.out_dir}/prism_anatomy_encoder_hh.pth')
torch.save(prism_h.style_encoder.state_dict(), f'{prism_h.out_dir}/prism_style_encoder_hh.pth')
torch.save(prism_h.decoder.state_dict(), f'{prism_h.out_dir}/prism_decoder_hh.pth')

# Site IOP
torch.save(prism_i.anatomy_encoder.state_dict(), f'{prism_i.out_dir}/prism_anatomy_encoder_iop.pth')
torch.save(prism_i.style_encoder.state_dict(), f'{prism_i.out_dir}/prism_style_encoder_iop.pth')
torch.save(prism_i.decoder.state_dict(), f'{prism_i.out_dir}/prism_decoder_iop.pth')

print(f'{'=' * 10} PRISM training ends {'=' * 10}')
