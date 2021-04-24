# INM705-Project-Capecchi---Kliafas-

This repository contains the solution to the INM705-Deep Learning for Image Analysis module at City, University of London.
The authors are Stelios Kliafas and Tommaso Capecchi.

## HOW TO RUN THE CODE

1- Download the virtual environment, the dataset and the pretrained models from the online repository https://cityuni-my.sharepoint.com/:f:/g/personal/tommaso_capecchi_city_ac_uk/EkwE6gmDjCVJl_jwx9bJ93sBXV_grgx1A-EuYPXYR8v4ng?e=RCUY3r

2- Insert the 'dataset' folder containing the 'lfw' folder with all the images in the root directory of the project.

3- To load properly the pretained models, after the 'checkpoint.pth' and the 'trained_model.pth' have been downloaded, just place them in the 'saved_models' folder. If the folder is not available just create it and place it in the root directory of the project.

4- To evaluate the model with the FID-Score metric, you need to download and install the pytorch-fid library, available at https://pypi.org/project/pytorch-fid/. To be able to run it smoothly without errors, please use the provided virtual environment.

5- Run the jupyter notebook.

