# Metadata For Original Silver Image Dataset 
Description: metadata for both the silver and non-silver class in CSV and JSON format

## Dataset Summary

Description: Metadata for both silver and non-silver classes, provided in CSV and JSON formats.

- **Total Images:** 20,564  
- **Silver Images:** 3,869  
- **Non-silver Images:** 16,695  

## Metadata Description

Metadata for the original dataset includes structured information for each image sample, such as:

- **image_index** – Unique identifier for each image  
- **label** – Class label (1 = silver, 0 = non-silver)  
- **mineral** – Mineral name associated with the image  
- **format** – Image file format (e.g., JPEG)  
- **file_size** – File size in bytes  
- **mode** – Color mode (e.g., RGB)  
- **image_width** – Image width in pixels  
- **image_height** – Image height in pixels  
- **aspect_ratio** – Ratio of width to height  

Some entries in the original dataset contain missing or invalid metadata values due to failed downloads or corrupted images. These inconsistencies were addressed during the data cleaning process to produce the final balanced dataset used for model training.
