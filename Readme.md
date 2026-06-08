
## General
This repository is for the paper Identity-Preserving GAN for Cross Spectral Iris Recognition. Follow this Readme to set up a dataset and run the training and testing in your own environment
## Dataset
To load a dataset, first ensure it is in the correct format. 

```
Dataset_name
For Training:
    NIR
	    class 1
		    NIR image 1
		    NIR image 2
		    ...
	    class 2
		    NIR image 1
		    NIR image 2
		    ...
	    ...

    VIS
	    class 1
		    VIS image 1
		    VIS image 2
		    ...
	    class 2
		    VIS image 1
		    VIS image 2
		    ...
	    ...
    ```
For Testing:
    NIR_Valid
	    class 1
		    NIR image 1
		    NIR image 2
		    ...
	    class 2
		    NIR image 1
		    NIR image 2
		    ...
	    ...

    VIS_Valid
	    class 1
		    VIS image 1
		    VIS image 2
		    ...
	    class 2
		    VIS image 1
		    VIS image 2
		    ...
	    ...
    ```
```
Ensure that each image in the NIR folder has a corresponding image in VIS folder, and there are the same amount of class folders for each. If the folder sizes are different the model will not work.

## Training

To train the GANs, first make sure to change this line to your dataset name (base folder shown in the dataset section above):

datasets_to_run=["Dataset"]

Ensure the config variable is set to the correct arguments, change these arguments to run in your own system:

- save_folder - Default: '/checkpoint/'
- Linux - True if running on Linux system, False if running on Windows
- modality - 'cropped' if using 256x256 iris images, 'normalized' if using 64x512 normalized iris images
- VIS_folder/NIR_folder - locations for each dataset relative to base directory
- model_name - What the model should be saved as

Once these arguments are correctly set, you can run the file "train_multispectral.py". There are more configuration options available in the top of the file, and feel free to change them for your own testing. Detailed explanations are given for each configuration, but feel free to reach out to me with any questions.

## Testing

To test the GANs, first make sure to change this line to your dataset name (base folder shown in the dataset section above):

datasets_to_run=["Dataset"]

Ensure the config variable is set to the correct arguments, change these arguments to run in your own system:

- Linux - True if running on Linux system, False if running on Windows
- modality - 'cropped' if using 256x256 iris images, 'normalized' if using 64x512 normalized iris images
- VIS_folder/NIR_folder - locations for each dataset relative to base directory
- checkpt_load - where the model checkpoint will be loaded from
- vistonir_dir/nirtovis_dir - where the translated test images will be saved

Once these arguments are correctly set, run the file "test_multispectral.py". The real NIR and VIS images, as well as the Synthetic images will be saved to your vistonir_dir and nirtovis_dir directories.

## Other files

The other files in this repository are helper files for the train/test functions
- Model.py --contains all models for the GAN
- Change_Params_Model.py - This is for changing the parameters for the classifier model
- dataset.py - handles the loading of the datasets
- utils.py - helper functions

If you have any questions about running the code, you can reach out to me by email and I will try to get back to you as soon as I can. 

hba0002@mix.wvu.edu

Thank you
