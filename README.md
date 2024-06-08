# Pose Adapted Shape Learning for Large-Pose Face Reenactment
![PASL.png](PASL.png)
> **Abstract:** We propose the Pose Adapted Shape Learning (PASL) for large-pose face reenactment. The PASL framework consists of three modules, namely the Pose-Adapted face Encoder (PAE), the Cycle-consistent Shape Generator (CSG), and the Attention-Embedded Generator (AEG). Different from previous approaches that use a single face encoder for identity preservation, we propose multiple Pose-Adapted face Encodes (PAEs) to better preserve facial identity across large poses.  Given a source face and a reference face, the CSG generates a recomposed shape that fuses the source identity and reference action in the shape space and meets the cycle consistency requirement. Taking the shape code and the source as inputs, the AEG learns the attention within the shape code and between the shape code and source style to enhance the generation of the desired target face. As existing benchmark datasets are inappropriate for evaluating large-pose face reenactment, we propose a scheme to compose large-pose face pairs and introduce the MPIE-LP (Large Pose) and VoxCeleb2-LP datasets as the new large-pose benchmarks. We compared our approach with state-of-the-art methods on MPIE-LP and VoxCeleb2-LP for large-pose performance and on VoxCeleb1 for the common scope of pose variation.


https://github.com/xxxxxx321/PASL/assets/151173571/88824fb8-fcaa-4035-a510-6e5cb1b9abd3







## Getting Started
- Clone the repo:
    ```
    git clone https://github.com/xxxxxx321/PASL
    cd PASL
    ```
## Installation
- Python 3.7
- Pytorch 1.12.1
2. Install the requirements
   ```
    conda env create -f environment.yml
    ```
3. Please refer to [Pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to install pytorch3d.

## Voxceleb2 LP Dataset
We offer the Voxceleb2 LP Dataset for download.
[GDrive](https://drive.google.com/drive/folders/1kHeXm9hOPCsF1Jyh9hVTqvPagYvvf-w8?usp=sharing)

## Pretrained Model
|Path|Description|
|---|---|
|[CSG Model](https://drive.google.com/file/d/10cNTvXIHllW1_rIgQovHE26_ASfKtLX7/view?usp=sharing)|Unzip it and place it into the data directory|
|[AEG Model](https://drive.google.com/file/d/1GCDhgMatmHH1LITpVgB_RTfpjAF13MXu/view?usp=sharing)|Unzip it and place it into the main directory|

## Auxiliary Models
|Path|Description|
|---|---|
|[Albedo model](https://drive.google.com/file/d/1VlSlEXAhseguor_T13Vy9oGpTgSakXZ8/view?usp=sharing)|Unzip it and place it into the data directory|
## Inference
```
    python demo_cam.py
    python demo_video.py
    python demo_ui.py
```
You can use `demo_cam.py` for a camera demo, or `demo_video.py` for a video demo. Additionally, we also offer a UI method using `demo_ui.py`.

## Validation
We provide six types of test lists: MPIE-LP, Voxceleb1, and Voxceleb2, including self-reenactment and cross-reenactment. Please note that after downloading, you need to change the paths of the pairs.

[Test list Sets](https://drive.google.com/file/d/10cNTvXIHllW1_rIgQovHE26_ASfKtLX7/view?usp=sharing)

The pretrained models for MPIE-LP, Voxceleb1, and Voxceleb2-LP can be downloaded from the following links.
|Pretrained Models|
|---|
|[MPIE-LP](https://drive.google.com/file/d/10cNTvXIHllW1_rIgQovHE26_ASfKtLX7/view?usp=sharing)|
|[Voxceleb1](https://drive.google.com/file/d/1GCDhgMatmHH1LITpVgB_RTfpjAF13MXu/view?usp=sharing)|
|[Voxceleb2-LP](https://drive.google.com/file/d/1GCDhgMatmHH1LITpVgB_RTfpjAF13MXu/view?usp=sharing)|

Please place the models for different datasets in the `./experiment` directory.

Next, use `main_lm_perceptual.py` to generate reenactment samples. The generated images will be placed in the `./expr/eval` directory.

```
    python main_lm_perceptual.py --dataset mpie 
```
For the `--dataset` parameter, please replace it as needed.

After generating the test samples, you can use `mean_poe_csim.py` and `mean_arcface_csim.py` to test CSIM. Please download the POE pretrained model and the ArcFace pretrained model from the following links, and extract them directly to start testing.

```
    python mean_poe_csim.py
    python mean_arcface_csim.py
```
    
