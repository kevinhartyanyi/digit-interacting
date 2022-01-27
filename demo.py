import pytorch_lightning as pl
from src.model.pl_factories import fetch_pl_model, DIGIT_PL
from src.model.pl_factories import fetch_val_dataloader
import elytra.torch_utils as torch_utils
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch.nn.functional as F
from src.utils.preprocessing import load_skeleton, generate_patch_image
import os.path as osp

from src.utils.vis import vis_keypoints, vis_3d_keypoints
from src.model.model import get_model_pl, Model

def demo_model(args):
    model = Model(args)
    model.load_pretrained("saved_models/db7cba8c1.pt")
    model.cuda()
    model.eval()

    transform = transforms.ToTensor()
    img_path = 'input.jpg'
    original_img = cv2.imread(img_path)
    img = original_img.copy()

    bbox = [69, 137, 165, 153] # xmin, ymin, width, height

    trans, scale, rot, do_flip, color_scale = [0,0], 1.0, 0.0, False, np.array([1,1,1])
    bbox[0] = bbox[0] + bbox[2] * trans[0]
    bbox[1] = bbox[1] + bbox[3] * trans[1]
    img, trans, inv_trans = generate_patch_image(img, bbox, do_flip, scale, rot, cfg.input_img_shape, cv2.INTER_LINEAR)
    img = np.clip(img * color_scale[None,None,:], 0, 255)

    img = transform(img.astype(np.float32))/255.0
    img = img[None,:,:,:]

    inputs = {"img": img.to('cuda:0')}
    targets = {}
    meta_info = {}

    with torch.no_grad():
        out = model(inputs, targets, meta_info, None)

    print("out", out)


    # joint set information is in annotations/skeleton.txt
    joint_num = 21 # single hand
    joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}
    skeleton = load_skeleton(osp.join('data/InterHand/annotations/skeleton.txt'), joint_num*2)

    img = img[0].cpu().numpy().transpose(1,2,0) # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
    joint_coord = out['joint_coord'][0].cpu().numpy() # x,y pixel, z root-relative discretized depth
    rel_root_depth = out['rel_root_depth'][0].cpu().numpy() # discretized depth
    hand_type = out['hand_type'][0].cpu().numpy() # handedness probability

    # restore joint coord to original image space and continuous depth space
    joint_coord[:,0] = joint_coord[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    joint_coord[:,1] = joint_coord[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    joint_coord[:,:2] = np.dot(inv_trans, np.concatenate((joint_coord[:,:2], np.ones_like(joint_coord[:,:1])),1).transpose(1,0)).transpose(1,0)
    joint_coord[:,2] = (joint_coord[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)

    # restore right hand-relative left hand depth to continuous depth space
    rel_root_depth = (rel_root_depth/cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root/2)

    # right hand root depth == 0, left hand root depth == rel_root_depth
    joint_coord[joint_type['left'],2] += rel_root_depth

    # handedness
    joint_valid = np.zeros((joint_num*2), dtype=np.float32)
    right_exist = False
    if hand_type[0] > 0.5: 
        right_exist = True
        joint_valid[joint_type['right']] = 1
    left_exist = False
    if hand_type[1] > 0.5:
        left_exist = True
        joint_valid[joint_type['left']] = 1

    print('Right hand exist: ' + str(right_exist) + ' Left hand exist: ' + str(left_exist))

    # visualize joint coord in 2D space
    filename = f"result_{img_path.split('.')[0]}.jpg"
    vis_img = original_img.copy()[:,:,::-1].transpose(2,0,1)
    vis_img = vis_keypoints(vis_img, joint_coord, joint_valid, skeleton, filename, vis_dir='.')


if __name__ == "__main__":
    from src.utils.config import cfg
    demo_model(cfg)
