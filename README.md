### Custom CNNs for Histopathology Classification Using the BreakHis Dataset  

This repository contains the code, documentation, and resources for building, training, and optimising custom Convolutional Neural Networks (CNNs) using the **BreakHis dataset** of histopathological slides. The project focuses on the classification of histopathology images into **benign** and **malignant** categories, tackling challenges like class imbalance, overfitting, and reproducibility.

#### Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

### Overview

Histopathology is critical in diagnosing diseases, but manual image analysis is both time-intensive and prone to variability. This project builds a robust pipeline to classify histopathology slides using the **BreakHis dataset**, which includes over 9,000 images at multiple magnification levels.  

#### Project Objectives:
1. Build, train, and optimise **custom CNN architectures** tailored for medical imaging.
2. Address **class imbalance** using techniques like weighted loss functions and data augmentation.
3. Evaluate models using detailed performance metrics such as accuracy, recall, precision, and F1-score.
4. Ensure reproducibility through fixed random seeds and dataset saving mechanisms.
5. Deploy the trained model using a Flask API for real-time predictions.

---

### Features
- **Dataset Pre-processing**:
  - Load and pre-process the BreakHis dataset.
  - Handle class imbalance with **class weights** and **oversampling**.
  - Apply **data augmentation** to improve model generalisation.
  
- **Custom CNN Architectures**:
  - Lightweight and optimised for computational efficiency.
  - Includes **dropout regularisation** to reduce overfitting.
  - Integrates **learning rate scheduling** for smoother convergence.

- **Comprehensive Evaluation**:
  - Generate confusion matrices and calculate key metrics.
  - Insights into false positives, false negatives, and model limitations.

- **Reproducibility and Efficiency**:
  - Save and reload preprocessed datasets using **Pickle**.
  - Fix random seeds to ensure consistent results.

- **Deployment-Ready Model**:
  - Save trained models in `.h5` format.
  - Deploy models as APIs using Flask for real-time predictions.

---

### Installation

#### Clone the Repository
```bash
git clone https://github.com/yourusername/custom-cnn-histopathology.git
cd custom-cnn-histopathology
```

#### Set Up the Environment
Create a virtual environment and install dependencies:

```bash
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

#### Requirements
Key libraries include:

 - `TensorFlow/Keras`
 - `NumPy`
 - `Scikit-learn`
 - `Flask`
 - `Matplotlib`

#### Install all dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### Usage
 - 1. Train the Model
Run the training script to preprocess the dataset, train the custom CNN, and save outputs:

```bash
python train.py
```
 - 2. Evaluate the Model
Generate confusion matrices and evaluate model performance:

```bash
python evaluate.py
```
 - 3. Deploy the Model
Start the Flask API to serve predictions:

```bash
python app.py
```
Upload a histopathology image via the API to classify it as benign or malignant.
```plaintext
custom-cnn-histopathology/
├── data/                   # Raw and preprocessed BreakHis data
├── models/                 # Saved models (.h5 files)
├── notebooks/              # Jupyter notebooks for experimentation
├── scripts/                # Python scripts for training, evaluation, and deployment
├── results/                # Confusion matrices and evaluation outputs
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── app.py                  # Flask application for model deployment

```
#### Technologies Used
 - `Python`: Programming language for data preprocessing, training, and evaluation.
 - `TensorFlow/Keras`: For building, training, and optimising custom CNNs.
 - `Scikit-learn`: For calculating class weights and generating evaluation metrics.
 - `Flask`: For deploying the model as an API.
 - `Matplotlib`: For visualising training performance and evaluation results.

---
### Results
Using the BreakHis dataset, the optimised pipeline achieved the following results:

 - **Accuracy**: 93.55%
 - **Precision**: 94.93%
 - **Recall (Sensitivity)**: 95.80%
 - **Specificity**: 88.52%
 - **F1-score**: 95.36%

---
### Future Work
 - Experiment with advanced architectures like `EfficientNet` or `Vision Transformers`.
 - Explore additional augmentation techniques to simulate diverse real-world conditions.
 - Incorporate explainability tools like `Grad-CAM` to visualise model decision-making.

---
### Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request. Ensure your code is documented and adheres to the project style.

---
### License
This project is licensed under the MIT License. See the LICENSE file for details.


