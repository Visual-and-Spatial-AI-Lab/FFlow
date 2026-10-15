import torch 
import torch.nn as nn
import torch.nn.functional as F
from .head import DPTHead

class DINOFeatureExtractor(nn.Module):
    def __init__(self,dino_model,out_dim=128):
        super(DINOFeatureExtractor,self).__init__()
        
        self.dino_model=dino_model
        self.out_dim=out_dim
        
        # 1x1 convolution to reduce feature dimensions from 384 -> 128
        self.feature_proj=nn.Conv2d(in_channels=384,out_channels=self.out_dim,kernel_size=1)

    def forward(self,x):
        _,_,h,w=x.shape

        # resize images before passing through DINOv2
        x=F.interpolate(x,size=(448,448),mode='bilinear',align_corners=False)

        # pass image/tensor through DINOv2
        with torch.no_grad():
            features=self.dino_model(x) # (B,1025,384)

        num_patches=features.shape[1] 
        patch_dim=int((num_patches-1)**0.5) # excluding the cls token

        assert patch_dim==32,f"Expected patch_dim=32 but got {patch_dim}"    
        
        features=features[:,1:,:].transpose(1,2)
        features=features.contiguous()
        features=features.view(x.shape[0],384,patch_dim,patch_dim)

        h_target=int(h/8)
        w_target=int(w/8)

        features_low_res=F.interpolate(features,size=(h_target,w_target),mode='bilinear',align_corners=False)
        features_high_res=F.interpolate(features,size=(h_target*2,w_target*2),mode='bilinear',align_corners=False)

        #reducing features from 384->128
        features_low_res=self.feature_proj(features_low_res)
        features_high_res=self.feature_proj(features_high_res)

        features_output=[features_low_res,features_high_res]

        return features_output
    

class DinoV2SFeature(nn.Module):
    def __init__(self,dino,features=128,lvl=-3):
        super().__init__()
        self.encoder=dino.eval()
        for p in self.encoder.parameters():
            p.requires_grad=False

        self.embed_dim=384
        self.intermediate_layer_idx=[2,5,8,11]

        self.dpt_head=DPTHead(
            in_channels=self.embed_dim,
            features=features,
            out_channels=[48,96,192,384],
            lvl=lvl
        )

    def forward(self,x):
        _,_,h,w=x.shape
        h_target=int(h/8)
        w_target=int(w/8)

        x=F.interpolate(x,size=(448,448),mode='bilinear',align_corners=False)
        B,C,H,W=x.shape

        patch_size=14
        patch_h,patch_w=H//patch_size,W//patch_size

        with torch.no_grad():
            out=self.encoder.dino_model(pixel_values=x,output_hidden_states=True)
            hs=out.hidden_states

        feats=[]

        for k in self.intermediate_layer_idx:
            t=hs[k+1]
            cls=t[:,:1,:]
            patches=t[:,1:,:]
            feats.append((patches,cls))

        outs=self.dpt_head(feats,patch_h,patch_w)
        features_low_res=F.interpolate(outs[0],size=(h_target,w_target),mode='bilinear',align_corners=False)
        features_high_res=F.interpolate(outs[0],size=(h_target*2,w_target*2),mode='bilinear',align_corners=False)
        return [features_low_res,features_high_res]         
    