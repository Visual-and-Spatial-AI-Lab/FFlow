from roco_spring_devkit.common.utils.registry import register_model
from roco_spring_devkit.optical_flow.models.base_model.base_model import BaseModel

from .fflow import FFlow


@register_model
class FFlow(BaseModel):

    def __init__(
        self,
        num_scales=2,
        upsample_factor=4,
        feature_channels=128,
        attention_type="swin",
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        num_head=1,
        image_size=(375, 1242),

        attn_splits_list=(2, 8),
        corr_radius_list=(-1, 4),
        prop_radius_list=(-1, 1),

        **kwargs,
    ):

        super().__init__(
            output_stride=32,
            loss_fn=None,
            **kwargs,
        )

        self.attn_splits_list = list(attn_splits_list)
        self.corr_radius_list = list(corr_radius_list)
        self.prop_radius_list = list(prop_radius_list)

        self.model = FFlow(
            num_scales=num_scales,
            upsample_factor=upsample_factor,
            feature_channels=feature_channels,
            attention_type=attention_type,
            num_transformer_layers=num_transformer_layers,
            ffn_dim_expansion=ffn_dim_expansion,
            num_head=num_head,
            image_size=list(image_size),
        )


    def forward(self, inputs):

        # -----------------------------------------------------
        # 1. RoCo input
        # [B, 2, 3, H, W]
        # -----------------------------------------------------

        images = inputs["images"]

        # -----------------------------------------------------
        # 2. Pad to model-compatible dimensions
        # -----------------------------------------------------

        # images, image_resizer = self.preprocess_images(
        #     images,
        #     resize_mode="pad",
        #     pad_mode="replicate",
        #     pad_two_side=False,
        # )
        images, image_resizer = self.preprocess_images(
            images,
            bgr_add=0.0,
            bgr_mult=1.0,
            bgr_to_rgb=False,
            stride=32,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        # -----------------------------------------------------
        # 3. Extract image pair
        # -----------------------------------------------------

        img0 = images[:, 0]   # [B, 3, H, W]
        img1 = images[:, 1]   # [B, 3, H, W]

        # =====================================================
        # FORWARD FLOW
        # img0 -> img1
        # =====================================================

        outputs_fw = self.model(
            img0,
            img1,
            attn_splits_list=self.attn_splits_list,
            corr_radius_list=self.corr_radius_list,
            prop_radius_list=self.prop_radius_list,
            pred_bidir_flow=False,
        )

        flow_fw = outputs_fw["flow_preds"][-1]

        # =====================================================
        # BACKWARD FLOW
        # img1 -> img0
        # =====================================================

        outputs_bw = self.model(
            img1,
            img0,
            attn_splits_list=self.attn_splits_list,
            corr_radius_list=self.corr_radius_list,
            prop_radius_list=self.prop_radius_list,
            pred_bidir_flow=False,
        )

        flow_bw = outputs_bw["flow_preds"][-1]

        # -----------------------------------------------------
        # 4. Remove padding
        # -----------------------------------------------------

        flow_fw = self.postprocess_predictions(
            flow_fw,
            image_resizer,
            is_flow=True,
        )

        flow_bw = self.postprocess_predictions(
            flow_bw,
            image_resizer,
            is_flow=True,
        )

        # -----------------------------------------------------
        # 5. Convert:
        #
        # [B, 2, H, W]
        #       ↓
        # [B, 1, 2, H, W]
        # -----------------------------------------------------

        flow_fw = flow_fw.unsqueeze(1)
        flow_bw = flow_bw.unsqueeze(1)

        # -----------------------------------------------------
        # 6. RoCo output
        # -----------------------------------------------------

        return {
            "flows": flow_fw,
            "flows_b": flow_bw,
        }