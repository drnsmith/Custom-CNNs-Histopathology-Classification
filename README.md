### Custom Convolutional Neural Networks for Histopathology Tumour Classification

#### **Overview**
This project focuses on developing, training, and optimizing custom **Convolutional Neural Networks (CNNs)** for the classification of histopathological images into benign and malignant categories. The study incorporates **stacking ensemble methods** to combine predictions from multiple CNN architectures for enhanced classification accuracy. The publicly available **BreakHis dataset** serves as the basis for evaluation.

---

#### **Motivation**
Histopathological analysis is critical for diagnosing diseases such as cancer. This project aims to:
1. Build lightweight CNN architectures tailored for histopathology images.
2. Address challenges like class imbalance and overfitting.
3. Evaluate the effectiveness of stacking ensemble methods in improving classification performance.

---

#### **Key Features**
- **Custom CNN Architectures**:
  - Lightweight, custom-designed CNNs optimized for medical imaging.
  - Dropout regularization and learning rate scheduling to reduce overfitting and ensure efficient training.

- **Stacking Ensemble Method**:
  - Combines predictions from:
    - Custom CNNs
    - Pre-trained models (e.g., ResNet, DenseNet)
  - Logistic regression is used as a meta-learner for ensemble integration.

- **Evaluation**:
  - Metrics include accuracy, precision, recall, F1-score, and ROC-AUC.
  - Confusion matrices for a detailed breakdown of performance.

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/drnsmith/Custom-CNNs-Histopathology-Classification.git
cd Custom-CNNs-Histopathology-Classification
```

### **2. Set Up the Environment**
Create a virtual environment and install the required dependencies:
```bash
python -m venv env
# Activate the virtual environment
# On Windows:
env\Scripts\activate
# On Unix or macOS:
source env/bin/activate
# Install dependencies
pip install -r requirements.txt
```

### **3. Download the BreakHis Dataset**
Obtain the BreakHis dataset from the official [BreakHis website](https://web.inf.ufpr.br/vri/breast-cancer-database) and place it in the `data/` directory.

---

## **Contributing**
Contributions are welcome! Follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -m 'Add NewFeature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Open a pull request.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
Special thanks to the creators of the **BreakHis dataset** and the researchers whose methodologies inspired this project.




