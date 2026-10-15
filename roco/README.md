# FFlow — ROCO Inference

This directory contains the custom FFlow model implementation for inference on the **RobustSpring benchmark**, integrated with the official [ROCO Spring DevKit](https://github.com/hmorimitsu/roco-spring-devkit).

The following instructions describe how to set up the inference environment, integrate the FFlow model, convert trained checkpoints into the ROCO-compatible format, and generate optical flow predictions.

## 1. Environment Setup

First, clone the official ROCO Spring DevKit:

```powershell
git clone https://github.com/hmorimitsu/roco-spring-devkit.git
cd roco-spring-devkit
```

### Create and activate a virtual environment

The following instructions are intended for **Windows PowerShell**, using Python 3.11.

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### Install PyTorch

Install PyTorch 2.0.1 and torchvision 0.15.2 with CUDA 11.8 support.

```powershell
python -m pip install torch==2.0.1 torchvision==0.15.2 `
    --index-url https://download.pytorch.org/whl/cu118
```

### Install additional dependencies

Install the Windows-specific dependencies using the `requirements_windows.txt` file provided in the FFlow repository.

Assuming FFlow and ROCO are located in the same parent directory:

```powershell
python -m pip install -r ..\FFlow\roco\requirements_windows.txt `
    --extra-index-url https://download.pytorch.org/whl/cu118
```

### Install ROCO in editable mode

```powershell
python -m pip install -e . --no-deps
```

The `--no-deps` flag prevents the original ROCO dependency declarations from overriding the custom Windows environment.

Verify the installation:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
python -m pip check
```

A correctly configured CUDA environment should return `True` for the first command.

## 2. Integrate the FFlow Model into ROCO

The custom FFlow implementation is provided under:

```text
FFlow/roco/optical_flow/models/
```

Replace the original ROCO models directory:

```text
roco-spring-devkit/roco_spring_devkit/optical_flow/models/
```

with the custom `optical_flow/models/` directory provided in this repository.

From the `roco-spring-devkit` root directory, run:

```powershell
# Back up the original models directory
Rename-Item .\roco_spring_devkit\optical_flow\models models_original

# Copy the custom FFlow models directory
Copy-Item ..\FFlow\roco\optical_flow\models `
    .\roco_spring_devkit\optical_flow\models -Recurse
```

After copying, the custom FFlow model should be available at:

```text
roco-spring-devkit/
└── roco_spring_devkit/
    └── optical_flow/
        └── models/
            └── ff_flow/
                └── roco_model.py
```

Ensure that the FFlow model is registered in the ROCO model registry, as described in the official [ROCO Optical Flow documentation](https://github.com/hmorimitsu/roco-spring-devkit/blob/main/roco_spring_devkit/optical_flow/README.md).

## 3. Convert the Trained Checkpoint

Before running inference, the trained FFlow checkpoint must be converted into the format expected by the ROCO evaluation framework.

Use the [checkpoint_roco_convert.py](https://github.com/Visual-and-Spatial-AI-Lab/FFlow/blob/main/checkpoint_roco_convert.py) script provided in the main FFlow repository.

The conversion script loads the original model weights and saves them in the ROCO-compatible checkpoint format.

### Prepare the checkpoint

Place the trained Spring checkpoint in the following directory:

```text
roco-spring-devkit/
└── roco_spring_devkit/
    └── spring/
        └── step_400000.pth
```

Create the `spring` directory if it does not already exist.

The conversion script uses the following input and output paths:

```python
src_path = r"..\spring\step_400000.pth"
dst_path = r"..\spring\step_400000_roco.pth"
```

Update these paths in the script if your checkpoint is stored in a different location.

### Run checkpoint conversion

Navigate to the ROCO optical flow directory from the `roco-spring-devkit` root:

```powershell
cd .\roco_spring_devkit\optical_flow
```

Run the conversion script:

```powershell
python ..\..\..\FFlow\checkpoint_roco_convert.py
```

The converted checkpoint will be saved at:

```text
roco_spring_devkit/spring/step_400000_roco.pth
```

**Note:** The conversion script uses paths relative to the current working directory. Execute it from `roco_spring_devkit/optical_flow/` when using the paths shown above.

## 4. Run RobustSpring Inference

Once the model is integrated and the checkpoint has been converted, inference can be performed using the official ROCO testing framework.

Make sure the dataset paths are configured in the ROCO `datasets.yaml` file, following the official ROCO documentation.

From the following working directory:

```text
roco-spring-devkit/roco_spring_devkit/optical_flow/
```

Run:

```powershell
python test.py `
    --data.test_dataset spring-robust `
    --model roco_spring_devkit.optical_flow.models.ff_flow.roco_model.FFlow `
    --ckpt_path "..\spring\step_400000_roco.pth" `
    --save_viz
```

### Command Arguments

| Argument | Description |
|---|---|
| `--data.test_dataset spring-robust` | Selects the RobustSpring test dataset. |
| `--model` | Specifies the custom FFlow model implementation. |
| `--ckpt_path` | Path to the converted ROCO-compatible checkpoint. |
| `--save_viz` | Enables saving optical flow visualizations. |

The command generates optical flow predictions for the RobustSpring benchmark using the trained FFlow model.

The generated predictions can be used for benchmark submission by following the official [ROCO Spring DevKit instructions](https://github.com/hmorimitsu/roco-spring-devkit).

## 5. Acknowledgements

Our inference and benchmark evaluation pipeline builds upon the [ROCO Spring DevKit](https://github.com/hmorimitsu/roco-spring-devkit). We thank the original authors for providing the evaluation framework and benchmark tools.
