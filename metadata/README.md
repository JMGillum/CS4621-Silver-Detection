## Metadata
All image classes have metadata stored in both CSV and JSON formats. The files are identical in content and follow the same labeling scheme. Each image has an associated set of metadata fields, as shown below.

<p align="left"> <img width="566" height="295" alt="image" src="https://github.com/user-attachments/assets/28f7c591-84c3-4af7-a736-714f61754659" /> </p>

Both formats store corresponding image data, where element 0 represents image_0 and element 1 represents image_1, continuing sequentially through the dataset. Metadata for silver images is stored in silver_metadata.csv and silver_metadata.json, while metadata for non-silver images is stored in non_silver_metadata.csv and non_silver_metadata.json. The `metadata/` folder contains structured information for all images in both the original and cleaned datasets. It is organized into two subfolders: <br><br>
<img width="335" height="263" alt="image" src="https://github.com/user-attachments/assets/d463924d-f703-4afc-9d2c-1edb6c93d1d8" />
- **original_dataset/**  
  Contains metadata for the initial unprocessed dataset. This dataset is unbalanced and includes all downloaded images prior to filtering. Metadata files provide class labels, mineral names, and image attributes, and may include inconsistencies due to mislabeled or corrupted data.

- **cleaned_dataset/**  
  Contains metadata for the processed and balanced dataset used for machine learning. This dataset has been cleaned to remove invalid samples and adjusted to achieve a balanced distribution between silver and non-silver classes. Metadata includes labels and additional image features used for model training.

- **README.md**  
  Provides documentation describing the metadata structure, file formats, and dataset organization.
