Best Models = PRE_denoise_oldcode_Final and noise_reduc_oldcode_Final
MAIN_DIR = ../../FUSEP

Workflow for training model:
  1. Use import os.py to preprocess and download data. Preprocessing includes noise reduction, scaling, and clipping the rgram to the range of second layer.
     Input: a "reloc" folder in MAIN_DIR containing the reloc file
     Output: rgrams in "rgram" folder, MAIN_DIR
     
  2. Use 01_matrix.py to create a ground truth label mask in the shape of the rgram
     Input: "reloc" and "rgram"
     Output: 0/1 labelled mask in "reloc_01" folder, MAIN_DIR
     
  3. Use txt_npy.py to convert txt to npy
     Input: "reloc_01" and "rgram"
     Output: .npy "reloc_01" and "rgram" in Data folder, MAIN_DIR
     
  4. Use cnn1d.py to train the model
     Input: "Data"
     Output: best.pt, final.pt, and best_threshold.npy in "model" folder, MAIN_DIR

  5. Use test.py to test the model on any rgram, and plot the result
     Input: "Model" and 1 "Data/rgram"
     Output: Image of Pre-denoised radargram (left), labelled relocation mask (center), and predicted mask overlay (right)

  at any point. Use plot_sharad.py to visualize files

Workflow for making predictions and map:
  1. 
  
