# BKAI-IGH-NeoPolyp-contest
Image Segmentation Model on BKAI-IGH-NeoPolyp contest

To use this model for inference, run the following script:

# Enviroment setup
Clone repository

```
git clone <https://github.com/sfatew/BKAI-IGH-NeoPolyp-contest.git>
cd <BKAI-IGH_NeoPolyp>
```

# Model Checkpoints

The model checkpoints `unet-model.pth` are stored in the drive <https://drive.google.com/drive/folders/1e4pmDVH3qFplgcQB5cd-DgE2VHKifs9p>

`unet-model.pth` is the model with the lowest valiadation loss during training.

### To run the model

put the model in the drive into the folder `/model`

You can run the following command to test on the image at your working directory

```
python3 infer.py --image_path path_to_image/image.jpeg --checkpoint model/model.pth
```
