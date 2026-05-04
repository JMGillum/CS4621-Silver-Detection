# Python Files Used To Build This Project
This project uses five Jupyter Notebooks to implement the data collection, validation, machine learning workflow for the silver detection dataset.

1. ***create_data.ipynb***

This notebook performs automated data acquisition to produce images for the silver and non-silver classes. It reads the Mindat image URL list, filters minerals into silver and non-silver categories, and programmatically downloads available JPEG images. The script includes error handling, logging, and class-balancing logic, including hard and easy negatives, to construct the final dataset.

2. ***data_exploration.ipynb***

This notebook explores the silver and non-silver dataset by generating distribution plots that show class balance, mineral types, and image characteristics such as width, aspect ratio, brightness, and contrast. It then prepares the dataset for machine learning by loading and formatting the images in Keras for CNN training.

3. ***machine_learning_models.ipynb***

This notebook applies five machine learning models to classify mineral images as silver or non silver using balanced datasets and structured preprocessing. The models include three CNN architectures, which are ResNet 50, DenseNet 121, and EfficientNet B0. They also include two transformer models, which are ViT Base Patch16 224 and DeiT Small Patch16 224. Each model is fine tuned and evaluated using standard metrics with threshold optimization to improve overall performance and maximize silver detection. The results are presented with graphs and performance metrics to support clear interpretation and comparison of model behavior.

4. ***metadata.ipynb***

This notebook builds and validates metadata for silver and non-silver image datasets used in training machine learning models. It verifies image files and extracts key metadata, including format, file size, color mode, dimensions, and aspect ratio. All metadata is organized and exported to JSON and CSV formats to support dataset validation and downstream machine learning workflows.

5. ***utility_JSON_conversion.ipynb***

This notebook provides a function to convert CSV files into JSON format.
This notebook a conversion function use to generate a JSON file from CSV file.

