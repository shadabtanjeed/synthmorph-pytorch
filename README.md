# SynthMorph

This repository provides a PyTorch implementation of SynthMorph for 3D medical image registration using synthetic training data.

Training is fully synthetic, while validation is performed on real data.
You can download the OASIS dataset from:
https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md
(use the full 6.6 GB version).

After downloading, run notebooks/eda.ipynb to create validation and test splits.

Update synthmorph/configs.py to configure:

- val_data_dir (path to validation patients)
- train/validation hyperparameters
- output paths and filenames

Expected validation folder structure:

- val_data_dir/
  - OASIS_OAS1_xxxx_MR1/
    - aligned_norm.nii.gz
    - aligned_seg35.nii.gz
  - OASIS_OAS1_xxxx_MR2/
    - aligned_norm.nii.gz
    - aligned_seg35.nii.gz

Training pipeline:

- Trains on synthetic data generated on the fly.
- Runs validation every 10 epochs on randomly paired real patients.
- Uses 26 labels for training and 35 labels for validation (label 0 ignored).
- Saves best model based on validation loss.
- Updates loss/dice plots at each validation step.

Run instructions:

1. Install dependencies:
   pip install -r requirements.txt
2. Set the validation path in synthmorph/configs.py.
3. Start training from repository root:
   python -m synthmorph.train
