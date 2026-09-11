import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import FeatureTransformer, FeatureFlowAttention
from .matching import global_correlation_softmax, local_correlation_softmax
from .geometry import flow_warp
from .utils import normalize_img, feature_add_position
from .dinomodel import DINOModel
from .dinobackbone import DINOFeatureExtractor
from .depth_anything_v2.dpt import DepthAnythingV2
from .feature_depth_processing import ResidualBlock,BasicConv 
from .downsample_proj import DownSampleProj 
from .dinobackbone import DinoV2SFeature


class FFlow(nn.Module):
    def __init__(self,
                 num_scales=1,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 image_size=[384,512],
                 dino_path="dino_path",
                 depth_model_path="depth_model_path",                 
                 **kwargs,
                 ):
        super(FFlow, self).__init__()

        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers
        self.image_size=image_size
        
        self.dino_path = dino_path
        self.depth_anything_v2_path = depth_model_path

        # DINOv2 backbone
        dino_model=DINOModel(model_name=self.dino_path)
        self.dinobackbone=DinoV2SFeature(dino_model)

        # Depth decoder
        self.dav2=DepthAnythingV2(encoder='vitb',features=128,out_channels=[96,192,384,768])
        self.dav2.load_state_dict(torch.load(self.depth_anything_v2_path,map_location='cpu'))
        self.dav2=self.dav2.eval()
        for param in self.dav2.parameters():
            param.requires_grad=False

        self.downsample_head=DownSampleProj(self.feature_channels)

        vit_feat_dim=128
        chans=[64,256]
        self.fusion_head = nn.Sequential(
        BasicConv(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, kernel_size=3, stride=1, padding=1, norm='instance'),
        ResidualBlock(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, norm_fn='instance'),
        ResidualBlock(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, norm_fn='instance'),
        )

        # projection layer 256->128
        self.proj_out=nn.Conv2d(256,self.feature_channels,kernel_size=1,stride=1)

        if self.num_scales>1:
            self.depth_out=nn.Conv2d(256,self.feature_channels,kernel_size=1,stride=1)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              attention_type=attention_type,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=feature_channels)

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
    
    def dino_extract_feature(self,img0,img1):
        concat=torch.cat((img0,img1),dim=0) # [2B,C,H,W]
        features=self.dinobackbone(concat)

        feature0, feature1= [], []

        for i in range(len(features)):
            feature=features[i]
            chunks=torch.chunk(feature,2,0)
            feature0.append(chunks[0])
            feature1.append(chunks[1])
        return feature0,feature1    

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      ):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h,
                                      self.upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow
    
    def min_max_scale(self,tensor,min_val=-5,max_val=5):
        min_t=tensor.amin(dim=1,keepdim=True)
        max_t=tensor.amax(dim=1,keepdim=True)
        scaled=(tensor-min_t)/(max_t-min_t+1e-8)
        scaled=scaled*(max_val -min_val)+min_val
        return scaled
    
    def forward(self, img0, img1,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                pred_bidir_flow=False,
                **kwargs,
                ):

        results_dict = {}
        flow_preds = []


        img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        with torch.no_grad():
            feature0_list,feature1_list=self.dino_extract_feature(img0,img1)

        # integrating depth and features
        image0_res = F.interpolate(img0, (518, 518), mode="bilinear", align_corners = False)
        image1_res = F.interpolate(img1, (518, 518), mode="bilinear", align_corners = False)

        images = torch.cat(
            [image0_res, image1_res],
            dim=0
        ).float()

        img_path1, depth = self.dav2(images)

        img0_path1, img1_path1 = img_path1.chunk(2, dim=0)

        height=img0.shape[2]
        width=img0.shape[3]

        im0_path1 = F.interpolate(img0_path1, (height,width), mode="bilinear", align_corners = False)
        im1_path1 = F.interpolate(img1_path1, (height,width), mode="bilinear", align_corners = False)                

        merged_input = torch.cat(
            [im0_path1, im1_path1],
            dim=0
        )

        merged_img4, merged_im = self.downsample_head(merged_input)

        img04_path1, img14_path1 = merged_img4.chunk(2, dim=0)
        im0_path1, im1_path1 = merged_im.chunk(2, dim=0)        

        feature0_list[0]=torch.cat((feature0_list[0],im0_path1),dim=1)
        feature1_list[0]=torch.cat((feature1_list[0],im1_path1),dim=1)

        batch_size = feature0_list[0].shape[0]

        features = torch.cat(
            [feature0_list[0], feature1_list[0]],
            dim=0
        )

        features = self.fusion_head(features)
        features = self.proj_out(features)

        feature0_list[0], feature1_list[0] = torch.split(
            features,
            batch_size,
            dim=0
        )

        if self.num_scales > 1:

            # depth_out together
            depth_features = torch.cat(
                [img04_path1, img14_path1],
                dim=0
            )

            depth_features = self.depth_out(depth_features)

            img04_path1, img14_path1 = depth_features.chunk(2, dim=0)

            # append depth features
            feature0_list[1] = torch.cat(
                [feature0_list[1], img04_path1],
                dim=1
            )

            feature1_list[1] = torch.cat(
                [feature1_list[1], img14_path1],
                dim=1
            )

            features = torch.cat(
                [feature0_list[1], feature1_list[1]],
                dim=0
            )

            features = self.proj_out(
                self.fusion_head(features)
            )

            feature0_list[1], feature1_list[1] = features.chunk(2, dim=0)
        flow = None

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
            else:  # local matching
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            # upsample to the original resolution for supervison
            if self.training:  # only need to upsample intermediate flow predictions at training time
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if pred_bidir_flow and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation
            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius)

            # bilinear upsampling at training time except the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                flow_preds.append(flow_up)

        results_dict.update({'flow_preds': flow_preds})

        return results_dict
