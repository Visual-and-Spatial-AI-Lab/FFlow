import os
import subprocess
import time



chairs_rf=[
"python", "main.py",
"--checkpoint_dir", "chairs",
"--stage", "chairs",
"--batch_size", "8",
"--num_steps", "200001",
"--val_dataset", "chairs","sintel",
"--output_path", "chairs",
"--lr", "4e-4",
"--image_size", "384", "512",
"--padding_factor", "32",
"--upsample_factor", "4",
"--num_scales", "2",
"--attn_splits_list", "2", "8",
"--corr_radius_list", "-1", "4",
"--prop_radius_list", "-1", "1",
"--with_speed_metric",
"--val_freq", "50000",
"--save_ckpt_freq", "50000"
"--dino_path", "facebook/dinov2-small",
"--depth_model_path", "depth_anything_v2_ckpt.pth"
]

things_rf=[
"python", "main.py",
"--checkpoint_dir", "things",
"--resume", "chairs/step_200000.pth",
"--stage", "things",
"--batch_size", "8",
"--num_steps", "360001",
"--val_dataset", "things", "sintel",
"--output_path", "things",
"--lr", "2e-4",
"--image_size", "384", "768",
"--padding_factor", "32",
"--upsample_factor", "4",
"--num_scales", "2",
"--attn_splits_list", "2", "8",
"--corr_radius_list", "-1", "4",
"--prop_radius_list", "-1", "1",
"--with_speed_metric",
"--val_freq", "40000",
"--save_ckpt_freq", "40000"
"--dino_path", "facebook/dinov2-small",
"--depth_model_path", "depth_anything_v2_ckpt.pth"
]

sintel_rf=[
"python", "main.py",
"--checkpoint_dir", "sintel",
"--resume", "things/step_360000.pth",
"--stage", "sintel",
"--batch_size", "4",
"--num_steps", "320001",
"--val_dataset", "sintel",
"--output_path", "sintel",
"--lr", "2e-4",
"--image_size", "320", "896",
"--padding_factor", "32",
"--upsample_factor", "4",
"--num_scales", "2",
"--attn_splits_list", "2", "8",
"--corr_radius_list", "-1", "4",
"--prop_radius_list", "-1", "1",
"--with_speed_metric",
"--val_freq", "20000",
"--save_ckpt_freq", "20000"
"--dino_path", "facebook/dinov2-small",
"--depth_model_path", "depth_anything_v2_ckpt.pth"
]


kitti_rf=[
"python", "main.py",
"--checkpoint_dir", "kitti",
"--resume", "sintel/step_320000.pth",
"--stage", "kitti",
"--batch_size", "4",
"--num_steps", "200001",
"--val_dataset", "kitti",
"--output_path", "kitti",
"--lr", "2e-4",
"--image_size", "320", "1152",
"--padding_factor", "32",
"--upsample_factor", "4",
"--num_scales", "2",
"--attn_splits_list", "2", "8",
"--corr_radius_list", "-1", "4",
"--prop_radius_list", "-1", "1",
"--with_speed_metric",
"--val_freq", "20000",
"--save_ckpt_freq", "20000"
"--dino_path", "facebook/dinov2-small",
"--depth_model_path", "depth_anything_v2_ckpt.pth"
]


spring_rf=[
"python", "main.py",
"--checkpoint_dir", "spring",
"--resume", "kitti/step_200000.pth",
"--stage", "spring",
"--batch_size", "4",
"--num_steps", "400001",
"--val_dataset", "sintel",
"--output_path", "spring",
"--lr", "1e-4",
"--image_size", "544", "960",
"--padding_factor", "32",
"--upsample_factor", "4",
"--num_scales", "2",
"--attn_splits_list", "2", "8",
"--corr_radius_list", "-1", "4",
"--prop_radius_list", "-1", "1",
"--with_speed_metric",
"--val_freq", "400000",
"--save_ckpt_freq", "60000"
"--dino_path", "facebook/dinov2-small",
"--depth_model_path", "depth_anything_v2_ckpt.pth"
]

env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

for i, cmd in enumerate([chairs_rf,things_rf,sintel_rf,kitti_rf,spring_rf], start=1):
    print("****** ---> ********* ", i, cmd)
    subprocess.run(cmd, shell=True, check=True, env=env)
    print("Starting Time delay ------")
    time.sleep(600)
    print("Ending Time delay")

print("All Done -------------")