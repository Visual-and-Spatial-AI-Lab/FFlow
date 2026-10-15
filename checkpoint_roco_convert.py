import torch

# Original checkpoint
src_path = r"..\spring\step_400000.pth"

# RoCo-compatible checkpoint
dst_path = r"..\spring\step_400000_roco.pth"


# ---------------------------------------------------------
# Load original checkpoint
# ---------------------------------------------------------
ckpt = torch.load(src_path, map_location="cpu")

state_dict = ckpt["model"]

print("Original checkpoint keys:")
for k in list(state_dict.keys())[:10]:
    print(k)


# ---------------------------------------------------------
# Add 'model.' prefix for TTSDPT wrapper
# ---------------------------------------------------------
roco_state_dict = {}

for key, value in state_dict.items():
    new_key = "model." + key
    roco_state_dict[new_key] = value


# ---------------------------------------------------------
# Create RoCo-compatible checkpoint
# ---------------------------------------------------------
roco_ckpt = {
    "state_dict": roco_state_dict
}


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
torch.save(roco_ckpt, dst_path)

print("\nSaved RoCo checkpoint to:")
print(dst_path)

print("\nConverted keys:")
for k in list(roco_state_dict.keys())[:10]:
    print(k)