import torch
import triton
import triton.language as tl


@triton.jit
def corr_forward_kernel(
    fmap1_ptr,
    fmap2_ptr,
    coords_ptr,
    corr_ptr,
    B,
    N,
    H1,
    W1,
    H2,
    W2,
    C,
    r,
    stride_f1_b,
    stride_f1_h,
    stride_f1_w,
    stride_f1_c,
    stride_f2_b,
    stride_f2_h,
    stride_f2_w,
    stride_f2_c,
    stride_c_b,
    stride_c_n,
    stride_c_h,
    stride_c_w,
    stride_c_2,
    stride_out_b,
    stride_out_n,
    stride_out_r,
    stride_out_h,
    stride_out_w,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pixels = H1 * W1
    num_spatial_blocks = tl.cdiv(num_pixels, BLOCK_HW)
    b = pid // num_spatial_blocks
    spatial_block_id = pid % num_spatial_blocks

    hw_offsets = spatial_block_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < num_pixels
    h1 = hw_offsets // W1
    w1 = hw_offsets % W1

    rd = 2 * r + 1

    for n in range(N):
        coord_x_ptr = (
            coords_ptr
            + b * stride_c_b
            + n * stride_c_n
            + h1 * stride_c_h
            + w1 * stride_c_w
            + 0 * stride_c_2
        )
        coord_y_ptr = (
            coords_ptr
            + b * stride_c_b
            + n * stride_c_n
            + h1 * stride_c_h
            + w1 * stride_c_w
            + 1 * stride_c_2
        )

        # tl.math.floor only accepts fp32/fp64. Keeping interpolation and the
        # dot product in fp32 also avoids accumulating fp16 rounding error.
        cx = tl.load(coord_x_ptr, mask=hw_mask, other=0.0).to(tl.float32)
        cy = tl.load(coord_y_ptr, mask=hw_mask, other=0.0).to(tl.float32)
        x0_base = tl.math.floor(cx)
        y0_base = tl.math.floor(cy)
        x0_base_int = x0_base.to(tl.int32)
        y0_base_int = y0_base.to(tl.int32)

        dx = cx - x0_base
        dy = cy - y0_base
        wa = ((1.0 - dx) * (1.0 - dy))[:, None]
        wb = (dx * (1.0 - dy))[:, None]
        wc = ((1.0 - dx) * dy)[:, None]
        wd = (dx * dy)[:, None]
        base_ptr2 = fmap2_ptr + b * stride_f2_b

        for iy in range(rd):
            for ix in range(rd):
                x0_int = x0_base_int + (ix - r)
                y0_int = y0_base_int + (iy - r)
                x1_int = x0_int + 1
                y1_int = y0_int + 1

                valid_a = (x0_int >= 0) & (x0_int < W2) & (y0_int >= 0) & (y0_int < H2)
                valid_b = (x1_int >= 0) & (x1_int < W2) & (y0_int >= 0) & (y0_int < H2)
                valid_c = (x0_int >= 0) & (x0_int < W2) & (y1_int >= 0) & (y1_int < H2)
                valid_d = (x1_int >= 0) & (x1_int < W2) & (y1_int >= 0) & (y1_int < H2)

                corr_val = tl.zeros([BLOCK_HW], dtype=tl.float32)
                for c_start in range(0, C, BLOCK_C):
                    c_offsets = c_start + tl.arange(0, BLOCK_C)
                    c_mask = c_offsets < C
                    mask_2d = hw_mask[:, None] & c_mask[None, :]

                    f1_ptrs = (
                        fmap1_ptr
                        + b * stride_f1_b
                        + h1[:, None] * stride_f1_h
                        + w1[:, None] * stride_f1_w
                        + c_offsets[None, :] * stride_f1_c
                    )
                    f1 = tl.load(f1_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

                    ptrs_a = (
                        base_ptr2
                        + y0_int[:, None] * stride_f2_h
                        + x0_int[:, None] * stride_f2_w
                        + c_offsets[None, :] * stride_f2_c
                    )
                    ptrs_b = (
                        base_ptr2
                        + y0_int[:, None] * stride_f2_h
                        + x1_int[:, None] * stride_f2_w
                        + c_offsets[None, :] * stride_f2_c
                    )
                    ptrs_c = (
                        base_ptr2
                        + y1_int[:, None] * stride_f2_h
                        + x0_int[:, None] * stride_f2_w
                        + c_offsets[None, :] * stride_f2_c
                    )
                    ptrs_d = (
                        base_ptr2
                        + y1_int[:, None] * stride_f2_h
                        + x1_int[:, None] * stride_f2_w
                        + c_offsets[None, :] * stride_f2_c
                    )

                    val_a = tl.load(
                        ptrs_a, mask=mask_2d & valid_a[:, None], other=0.0
                    ).to(tl.float32)
                    val_b = tl.load(
                        ptrs_b, mask=mask_2d & valid_b[:, None], other=0.0
                    ).to(tl.float32)
                    val_c = tl.load(
                        ptrs_c, mask=mask_2d & valid_c[:, None], other=0.0
                    ).to(tl.float32)
                    val_d = tl.load(
                        ptrs_d, mask=mask_2d & valid_d[:, None], other=0.0
                    ).to(tl.float32)

                    f2 = val_a * wa + val_b * wb + val_c * wc + val_d * wd
                    corr_val += tl.sum(f1 * f2, axis=1)

                idx_r = ix * rd + iy
                out_ptr = (
                    corr_ptr
                    + b * stride_out_b
                    + n * stride_out_n
                    + idx_r * stride_out_r
                    + h1 * stride_out_h
                    + w1 * stride_out_w
                )
                tl.store(out_ptr, corr_val, mask=hw_mask)


@triton.jit
def corr_pyramid_forward_kernel(
    fmap1_ptr,
    fmap2_packed_ptr,
    coords_ptr,
    fmap2_offsets_ptr,
    fmap2_shapes_ptr,
    level_scales_ptr,
    corr_ptr,
    B,
    H1,
    W1,
    C,
    r,
    stride_f1_b,
    stride_f1_h,
    stride_f1_w,
    stride_f1_c,
    stride_c_b,
    stride_c_2,
    stride_c_h,
    stride_c_w,
    stride_out_b,
    stride_out_level,
    stride_out_r,
    stride_out_h,
    stride_out_w,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    level = tl.program_id(1)

    num_pixels = H1 * W1
    num_spatial_blocks = tl.cdiv(num_pixels, BLOCK_HW)
    b = pid // num_spatial_blocks
    spatial_block_id = pid % num_spatial_blocks

    hw_offsets = spatial_block_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < num_pixels
    h1 = hw_offsets // W1
    w1 = hw_offsets % W1

    h2 = tl.load(fmap2_shapes_ptr + 2 * level)
    w2 = tl.load(fmap2_shapes_ptr + 2 * level + 1)
    fmap2_offset = tl.load(fmap2_offsets_ptr + level)
    level_scale = tl.load(level_scales_ptr + level)
    fmap2_base = fmap2_packed_ptr + fmap2_offset + b * h2 * w2 * C

    cx = (
        tl.load(
            coords_ptr + b * stride_c_b + h1 * stride_c_h + w1 * stride_c_w,
            mask=hw_mask,
            other=0.0,
        ).to(tl.float32)
        * level_scale
    )
    cy = (
        tl.load(
            coords_ptr
            + b * stride_c_b
            + stride_c_2
            + h1 * stride_c_h
            + w1 * stride_c_w,
            mask=hw_mask,
            other=0.0,
        ).to(tl.float32)
        * level_scale
    )
    x0_base = tl.math.floor(cx)
    y0_base = tl.math.floor(cy)
    x0_base_int = x0_base.to(tl.int32)
    y0_base_int = y0_base.to(tl.int32)

    dx = cx - x0_base
    dy = cy - y0_base
    wa = ((1.0 - dx) * (1.0 - dy))[:, None]
    wb = (dx * (1.0 - dy))[:, None]
    wc = ((1.0 - dx) * dy)[:, None]
    wd = (dx * dy)[:, None]

    rd = 2 * r + 1
    for iy in range(rd):
        for ix in range(rd):
            x0_int = x0_base_int + (ix - r)
            y0_int = y0_base_int + (iy - r)
            x1_int = x0_int + 1
            y1_int = y0_int + 1

            valid_a = (x0_int >= 0) & (x0_int < w2) & (y0_int >= 0) & (y0_int < h2)
            valid_b = (x1_int >= 0) & (x1_int < w2) & (y0_int >= 0) & (y0_int < h2)
            valid_c = (x0_int >= 0) & (x0_int < w2) & (y1_int >= 0) & (y1_int < h2)
            valid_d = (x1_int >= 0) & (x1_int < w2) & (y1_int >= 0) & (y1_int < h2)

            corr_val = tl.zeros([BLOCK_HW], dtype=tl.float32)
            for c_start in range(0, C, BLOCK_C):
                c_offsets = c_start + tl.arange(0, BLOCK_C)
                c_mask = c_offsets < C
                mask_2d = hw_mask[:, None] & c_mask[None, :]

                f1_ptrs = (
                    fmap1_ptr
                    + b * stride_f1_b
                    + h1[:, None] * stride_f1_h
                    + w1[:, None] * stride_f1_w
                    + c_offsets[None, :] * stride_f1_c
                )
                f1 = tl.load(f1_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

                ptrs_a = (
                    fmap2_base
                    + (y0_int[:, None] * w2 + x0_int[:, None]) * C
                    + c_offsets[None, :]
                )
                ptrs_b = (
                    fmap2_base
                    + (y0_int[:, None] * w2 + x1_int[:, None]) * C
                    + c_offsets[None, :]
                )
                ptrs_c = (
                    fmap2_base
                    + (y1_int[:, None] * w2 + x0_int[:, None]) * C
                    + c_offsets[None, :]
                )
                ptrs_d = (
                    fmap2_base
                    + (y1_int[:, None] * w2 + x1_int[:, None]) * C
                    + c_offsets[None, :]
                )

                val_a = tl.load(ptrs_a, mask=mask_2d & valid_a[:, None], other=0.0).to(
                    tl.float32
                )
                val_b = tl.load(ptrs_b, mask=mask_2d & valid_b[:, None], other=0.0).to(
                    tl.float32
                )
                val_c = tl.load(ptrs_c, mask=mask_2d & valid_c[:, None], other=0.0).to(
                    tl.float32
                )
                val_d = tl.load(ptrs_d, mask=mask_2d & valid_d[:, None], other=0.0).to(
                    tl.float32
                )

                f2 = val_a * wa + val_b * wb + val_c * wc + val_d * wd
                corr_val += tl.sum(f1 * f2, axis=1)

            idx_r = ix * rd + iy
            out_ptr = (
                corr_ptr
                + b * stride_out_b
                + level * stride_out_level
                + idx_r * stride_out_r
                + h1 * stride_out_h
                + w1 * stride_out_w
            )
            tl.store(out_ptr, corr_val, mask=hw_mask)


@triton.jit
def corr_backward_kernel(
    fmap1_ptr,
    fmap2_ptr,
    coords_ptr,
    corr_grad_ptr,
    fmap1_grad_ptr,
    fmap2_grad_ptr,
    B,
    N,
    H1,
    W1,
    H2,
    W2,
    C,
    r,
    stride_f1_b,
    stride_f1_h,
    stride_f1_w,
    stride_f1_c,
    stride_f2_b,
    stride_f2_h,
    stride_f2_w,
    stride_f2_c,
    stride_c_b,
    stride_c_n,
    stride_c_h,
    stride_c_w,
    stride_c_2,
    stride_cg_b,
    stride_cg_n,
    stride_cg_r,
    stride_cg_h,
    stride_cg_w,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pixels = H1 * W1
    b = pid // num_pixels
    hw = pid % num_pixels
    h1 = hw // W1
    w1 = hw % W1

    c_offsets = tl.arange(0, BLOCK_C)
    valid_channels = c_offsets < C

    f1_ptrs = (
        fmap1_ptr
        + b * stride_f1_b
        + h1 * stride_f1_h
        + w1 * stride_f1_w
        + c_offsets * stride_f1_c
    )
    f1 = tl.load(f1_ptrs, mask=valid_channels, other=0.0).to(tl.float32)

    f1_grad = tl.zeros([BLOCK_C], dtype=tl.float32)
    rd = 2 * r + 1

    for n in range(N):
        coord_x_ptr = (
            coords_ptr
            + b * stride_c_b
            + n * stride_c_n
            + h1 * stride_c_h
            + w1 * stride_c_w
            + 0 * stride_c_2
        )
        coord_y_ptr = (
            coords_ptr
            + b * stride_c_b
            + n * stride_c_n
            + h1 * stride_c_h
            + w1 * stride_c_w
            + 1 * stride_c_2
        )

        cx = tl.load(coord_x_ptr).to(tl.float32)
        cy = tl.load(coord_y_ptr).to(tl.float32)
        x0_base = tl.math.floor(cx)
        y0_base = tl.math.floor(cy)
        x0_base_int = x0_base.to(tl.int32)
        y0_base_int = y0_base.to(tl.int32)

        dx = cx - x0_base
        dy = cy - y0_base
        wa = (1.0 - dx) * (1.0 - dy)
        wb = dx * (1.0 - dy)
        wc = (1.0 - dx) * dy
        wd = dx * dy
        base_ptr2 = fmap2_ptr + b * stride_f2_b
        base_ptr2_grad = fmap2_grad_ptr + b * stride_f2_b

        for iy in range(rd):
            for ix in range(rd):
                x0_int = x0_base_int + (ix - r)
                y0_int = y0_base_int + (iy - r)
                x1_int = x0_int + 1
                y1_int = y0_int + 1

                valid_a = (x0_int >= 0) & (x0_int < W2) & (y0_int >= 0) & (y0_int < H2)
                valid_b = (x1_int >= 0) & (x1_int < W2) & (y0_int >= 0) & (y0_int < H2)
                valid_c = (x0_int >= 0) & (x0_int < W2) & (y1_int >= 0) & (y1_int < H2)
                valid_d = (x1_int >= 0) & (x1_int < W2) & (y1_int >= 0) & (y1_int < H2)

                ptrs_a = (
                    base_ptr2
                    + y0_int * stride_f2_h
                    + x0_int * stride_f2_w
                    + c_offsets * stride_f2_c
                )
                ptrs_b = (
                    base_ptr2
                    + y0_int * stride_f2_h
                    + x1_int * stride_f2_w
                    + c_offsets * stride_f2_c
                )
                ptrs_c = (
                    base_ptr2
                    + y1_int * stride_f2_h
                    + x0_int * stride_f2_w
                    + c_offsets * stride_f2_c
                )
                ptrs_d = (
                    base_ptr2
                    + y1_int * stride_f2_h
                    + x1_int * stride_f2_w
                    + c_offsets * stride_f2_c
                )

                val_a = tl.load(ptrs_a, mask=valid_channels & valid_a, other=0.0).to(
                    tl.float32
                )
                val_b = tl.load(ptrs_b, mask=valid_channels & valid_b, other=0.0).to(
                    tl.float32
                )
                val_c = tl.load(ptrs_c, mask=valid_channels & valid_c, other=0.0).to(
                    tl.float32
                )
                val_d = tl.load(ptrs_d, mask=valid_channels & valid_d, other=0.0).to(
                    tl.float32
                )

                idx_r = ix * rd + iy
                grad_ptr = (
                    corr_grad_ptr
                    + b * stride_cg_b
                    + n * stride_cg_n
                    + idx_r * stride_cg_r
                    + h1 * stride_cg_h
                    + w1 * stride_cg_w
                )
                g = tl.load(grad_ptr).to(tl.float32)

                # Accumulate f1 gradients
                f2 = wa * val_a + wb * val_b + wc * val_c + wd * val_d
                f1_grad += g * f2

                # Scatter f2 gradients via atomic additions
                gf1 = g * f1
                tl.atomic_add(
                    base_ptr2_grad
                    + y0_int * stride_f2_h
                    + x0_int * stride_f2_w
                    + c_offsets * stride_f2_c,
                    gf1 * wa,
                    mask=valid_channels & valid_a,
                )
                tl.atomic_add(
                    base_ptr2_grad
                    + y0_int * stride_f2_h
                    + x1_int * stride_f2_w
                    + c_offsets * stride_f2_c,
                    gf1 * wb,
                    mask=valid_channels & valid_b,
                )
                tl.atomic_add(
                    base_ptr2_grad
                    + y1_int * stride_f2_h
                    + x0_int * stride_f2_w
                    + c_offsets * stride_f2_c,
                    gf1 * wc,
                    mask=valid_channels & valid_c,
                )
                tl.atomic_add(
                    base_ptr2_grad
                    + y1_int * stride_f2_h
                    + x1_int * stride_f2_w
                    + c_offsets * stride_f2_c,
                    gf1 * wd,
                    mask=valid_channels & valid_d,
                )

    # Store final f1_grad at the end
    f1_grad_ptrs = (
        fmap1_grad_ptr
        + b * stride_f1_b
        + h1 * stride_f1_h
        + w1 * stride_f1_w
        + c_offsets * stride_f1_c
    )
    tl.store(f1_grad_ptrs, f1_grad, mask=valid_channels)


class TritonCorr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2_i, coords, r):
        if torch.onnx.is_in_onnx_export():
            batch_size, height_1, width_1, _ = fmap1.shape
            _, num_slices, _, _, _ = coords.shape
            radius = int(r)
            window_size = 2 * radius + 1
            corr = torch.empty(
                (batch_size, num_slices, window_size * window_size, height_1, width_1),
                device=fmap1.device,
                dtype=fmap1.dtype,
            )
            return (corr,)

        ctx.save_for_backward(fmap1, fmap2_i, coords)
        ctx.r = r

        B, H1, W1, C = fmap1.shape
        _, N, _, _, _ = coords.shape
        _, H2, W2, _ = fmap2_i.shape
        B = int(B)
        H1 = int(H1)
        W1 = int(W1)
        C = int(C)
        N = int(N)
        H2 = int(H2)
        W2 = int(W2)
        r = int(r)

        rd = 2 * r + 1
        corr = torch.empty(
            (B, N, rd * rd, H1, W1), device=fmap1.device, dtype=fmap1.dtype
        )

        BLOCK_HW = 16
        BLOCK_C = min(64, triton.next_power_of_2(C))
        grid = (B * triton.cdiv(H1 * W1, BLOCK_HW),)

        corr_forward_kernel[grid](
            fmap1,
            fmap2_i,
            coords,
            corr,
            B,
            N,
            H1,
            W1,
            H2,
            W2,
            C,
            r,
            fmap1.stride(0),
            fmap1.stride(1),
            fmap1.stride(2),
            fmap1.stride(3),
            fmap2_i.stride(0),
            fmap2_i.stride(1),
            fmap2_i.stride(2),
            fmap2_i.stride(3),
            coords.stride(0),
            coords.stride(1),
            coords.stride(2),
            coords.stride(3),
            coords.stride(4),
            corr.stride(0),
            corr.stride(1),
            corr.stride(2),
            corr.stride(3),
            corr.stride(4),
            BLOCK_HW=BLOCK_HW,
            BLOCK_C=BLOCK_C,
        )
        return corr

    @staticmethod
    def backward(ctx, corr_grad):
        fmap1, fmap2_i, coords = ctx.saved_tensors
        r = int(ctx.r)

        corr_grad = corr_grad.contiguous()

        B, H1, W1, C = fmap1.shape
        _, N, _, _, _ = coords.shape
        _, H2, W2, _ = fmap2_i.shape
        B = int(B)
        H1 = int(H1)
        W1 = int(W1)
        C = int(C)
        N = int(N)
        H2 = int(H2)
        W2 = int(W2)

        fmap1_grad = torch.empty_like(fmap1)
        fmap2_grad = torch.zeros_like(fmap2_i)

        # The RAFT architecture stops gradients at the coords node during iterative updates.
        # We can return a tensor of zeros to align with original autograd footprints.
        coords_grad = torch.zeros_like(coords)

        BLOCK_C = triton.next_power_of_2(C)
        grid = (B * H1 * W1,)

        corr_backward_kernel[grid](
            fmap1,
            fmap2_i,
            coords,
            corr_grad,
            fmap1_grad,
            fmap2_grad,
            B,
            N,
            H1,
            W1,
            H2,
            W2,
            C,
            r,
            fmap1.stride(0),
            fmap1.stride(1),
            fmap1.stride(2),
            fmap1.stride(3),
            fmap2_i.stride(0),
            fmap2_i.stride(1),
            fmap2_i.stride(2),
            fmap2_i.stride(3),
            coords.stride(0),
            coords.stride(1),
            coords.stride(2),
            coords.stride(3),
            coords.stride(4),
            corr_grad.stride(0),
            corr_grad.stride(1),
            corr_grad.stride(2),
            corr_grad.stride(3),
            corr_grad.stride(4),
            BLOCK_C=BLOCK_C,
        )

        return fmap1_grad, fmap2_grad, coords_grad, None

    @staticmethod
    def symbolic(g, fmap1, fmap2_i, coords, r):
        # This writes "custom_ops::TritonCorr" into the ONNX file
        node = g.op("custom_ops::TritonCorr", fmap1, fmap2_i, coords, r_i=r)

        fmap1_sizes = fmap1.type().sizes()
        coords_sizes = coords.type().sizes()
        if fmap1_sizes is not None and coords_sizes is not None:
            radius = int(r)
            window_size = 2 * radius + 1
            output_sizes = [
                coords_sizes[0],
                coords_sizes[1],
                window_size * window_size,
                fmap1_sizes[1],
                fmap1_sizes[2],
            ]
            node.setType(fmap1.type().with_sizes(output_sizes))

        return node
