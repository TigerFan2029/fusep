Best Models = PRE_denoise_oldcode_Final and noise_reduc_oldcode_Final

MAIN_DIR = ../../FUSEP

"" = folder


Workflow for training model:

  1. Use import os.py to preprocess and download data according to "reloc". Preprocessing includes noise reduction, scaling, and clipping the rgram to the range of second layer.
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

  * at any point. Use plot_sharad.py to visualize files


Workflow for making predictions and map:

  1. Use download_rgram.py to preprocess and download data according to .csv file from GIS. Preprocessing includes noise reduction and scaling.
     Input: a .csv file in MAIN_DIR containing the id of the wanted rgrams
     Output: rgrams in "rgram_full" folder, MAIN_DIR

  2. (optional) Use clip_rgram.py to clip the "rgram_full".
     Input: .csv file, clipped.csv file, and "rgram_full". *clipped.csv is the clipped .csv file, done in GIS software, change_coor_qgis.py may help in finding the new coordinates in GIS.
     Output: clipped rgrams in "rgram_clipped" folder, MAIN_DIR

  3. Use predict.py to run the model on all "rgram_full" or "rgram_clipped", and identify layers.
     Input: "models", .csv file, "rgram_full" or "rgram_clipped".
     Output: In "predictions" folder, MAIN_DIR: .csv files containing the data, including time-delay between layer 1 and 2 for all segments. .txt files of predicted mask. Plots of Denoised              radargram (left), predicted mask (center), and segmented layers (right).

  4. Use mapping.py to convert the final .csv files of the data to a .shp file.
     Input: .csv files in "predictions"
     Output: .shp file to open in GIS software


       
