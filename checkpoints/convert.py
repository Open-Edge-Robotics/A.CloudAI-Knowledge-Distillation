import torch

checkpoint = torch.load('simmim_vit_large_sz224_8xb128_accu2_step_fp16_ep800_full.pth')
checkpoint = checkpoint['state_dict']
print(checkpoint.keys())

new_ckpt = dict()

for key in checkpoint.keys():
    new_ckpt[key.replace('backbone.', '')] = checkpoint[key]

torch.save(new_ckpt, 'simmim_vit_large_sz224_8xb128_accu2_step_fp16_ep800_full_.pth')
