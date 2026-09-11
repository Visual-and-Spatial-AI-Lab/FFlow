import torch 
import torch.nn as nn
from transformers import Dinov2Model

class DINOModel(nn.Module):
    def __init__(self,model_name,trainable_layer=None):
        super(DINOModel,self).__init__()

        self.dino_model=Dinov2Model.from_pretrained(model_name)

        self.dino_model.eval()

        for param in self.dino_model.parameters():
            param.requires_grad=False

    def forward_features(self,x):
        # extract patch embeddings from DINOv2
        with torch.no_grad():
            outputs=self.dino_model(x)
        return outputs.last_hidden_state #(B,num_patches,hidden_dim)

    def forward(self,x):
        return self.forward_features(x)

#model_path="/Users/praroopchanda/Desktop/gmflow_depth_depth_vitb_conv4/modelsfacebookdinov2small/snapshots/e"                