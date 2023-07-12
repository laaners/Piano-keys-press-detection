# Folder Structure
```bash
.
|   keys_map.json
|   main.ipynb
|   readme.md
|   utils.py
|   
+---final_results
|   |   covered_bolero.mp4
|   |   
|   +---audio
|   |       covered_bolero.m4a
|   |       unet_bolero.m4a
|   |       
|   +---json
|   |       covered_bolero.json
|   |       unet_bolero.json
|   |       
|   +---midi
|   |       covered_bolero.mid
|   |       unet_bolero.mid
|   |       
|   \---videos_w_detected_keys
|           covered_bolero.mp4
|           unet_bolero.mp4
|           
+---pdf_ppt
|       F05_Paesano_Hu.pdf
|       F05_Paesano_Hu.pptx
|       
+---segmentation
|   |   segmentation_unet.ipynb
|   |   
|   +---processed_training_labels
|   |       aboveall__6000.0.png
|   |       ...
|   |       piano_bolero__960.0.png
|   |       ...
|   |       
|   +---training_data
|   |       aboveall__6000.0.png
|   |       ...
|   |       piano_bolero__960.0.png
|   |       ...
|   |       
|   \---UNet_3_classes
|       |   keras_metadata.pb
|       |   saved_model.pb
|       |   
|       +---assets
|       \---variables
|               variables.data-00000-of-00001
|               variables.index
|               
\---videos
    \---piano
            covered_bolero.mp4
            ...            
```

# Instructions to Run

The [main.ipynb](main.ipynb) notebook can be executed all to test on [covered_bolero.mp4](videos/piano/covered_bolero.mp4). Cells are titled according to the operations that are being performed, it's advisable to execute cells one by one, the following cells are alternative to the other: "Automatic" and "Manual" under the `Keyboard rectification` section.

The [main.ipynb](main.ipynb) notebook uses:
- The functions defined in [utils.py](utils.py) and will save the results in the folder where the input video is located (so for our example it will save in `videos/piano/`)
- The file [keys_map.json](keys_map.json) which contains midi keys ids

## U-Net imports
The [segmentation_unet.ipynb](segmentation/segmentation_unet.ipynb) notebook trains a U-Net network aimed at segmentation.

The train set is divided into the original frames and the corresponding masks, located respectively in `segmentation\training_data`  and `segmentation\processed_training_labels`.

The notebook has already been run and the model has already been trained, specifically it is saved in the `segmentation/UNet_3_classes` folder and it is imported into the [main.ipynb](main.ipynb) notebook by:
```python
import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers

model = tfk.models.load_model('segmentation/UNet_3_classes', custom_objects={'UpdatedMeanIoU': tf.keras.metrics.MeanIoU})
```

## Saved experimental results
The experimental results were all moved into the `final_results` folder: the notebook outputs the detected keys in a `.json` file and then in a `.mid` and by configuring the function 
```python
utils.playVideo(..., save_video=True, video_name=..., ...)
```
the notebook will also generate an output video with the detected pressed keys bounding boxes drawn, an example:

https://github.com/laaners/POLIMI_IACV_PROJECT/assets/75478979/d19e68b0-fdbe-41e6-b1aa-b4955fb8bbd0

## Detailed report
The pdf report and the ppt presentation are located in `pdf_ppt`, the [F05_Paesano_Hu.pdf](pdf_ppt/F05_Paesano_Hu.pdf) explains in detail the algorithm and reasoning applied

