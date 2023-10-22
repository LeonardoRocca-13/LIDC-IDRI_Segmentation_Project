# Lung Cancer Detection using Deep Learning

This project aims to develop a deep learning model for lung cancer detection using Fully Convolutional Neural Networks (FCNN) and U-NET on the LIDC-IDRI dataset. The LIDC-IDRI dataset is a collection of CT scan images of the lungs, which provides annotations for the presence of lung nodules.

## Objective

The objective of this project is to build an accurate and reliable model that can detect lung nodules from CT scan images and classify them as cancerous or non-cancerous. Early detection of lung cancer is crucial for effective treatment, and the use of deep learning algorithms can aid in the early detection process.

## Methodology

1. **Data Preprocessing**: The LIDC-IDRI dataset will be preprocessed to ensure consistent voxel spacing, segment the lung region, and normalize pixel values. This preprocessing step is crucial for preparing the dataset for model training.

2. **Model Architecture**: A Fully Convolutional Neural Network (FCNN) will be used for lung cancer segmentation. The model will consist of multiple convolutional layers for feature extraction, followed by trasposed convolutional layers to resize the image to the full scale. Finally another convolution, and a sigmoid activation function will be used for predicting the presence or absence of lung nodules.
In addition, the U-NET Model will be used as it is the latest cutting edge tecnology in terms of bio-medical segmentation.

4. **Training and Evaluation**: The model will be trained of the LIDC-IDRI dataset and evaluated on a separate testing set. Training techniques such as data augmentation, batch normalization, and early stopping will be employed to improve the model's performance and prevent overfitting. Performance metrics such as accuracy, precision, recall, and F1 score will be used to evaluate the model's performance.

5. **Further Enhancements**: To enhance the model's performance, additional techniques such as transfer learning, ensemble learning, or more advanced architectures like 3D FCNN or attention-based models can be explored. Incorporating other clinical data or features can also be considered to improve the accuracy of lung cancer detection.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies mentioned in the `requirements.txt` file.
3. Preprocess the LIDC-IDRI dataset using the provided scripts.
4. Train the deep learning model using the preprocessed dataset.
5. Evaluate the model's performance on the testing set.
6. Experiment with different techniques and architectures to further improve the model's accuracy.

## Dataset

The LIDC-IDRI dataset can be downloaded from [link to dataset]([https://example.com](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)). Please refer to the dataset's documentation for more information on usage and citation requirements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug fixes, please open an issue or submit a pull request.

## Acknowledgements

We would like to acknowledge the creators of the LIDC-IDRI dataset for providing this valuable resource for research and development in lung cancer detection.

## Contact

For any questions or inquiries, please contact leo.rocca03@outlook.it


 
