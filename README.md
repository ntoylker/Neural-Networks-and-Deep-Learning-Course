# Neural-Networks-and-Deep-Learning-Course
# 🧠 Assignment 1 – Neural Networks & Deep Learning

## 🎯 Objective
The goal of this assignment is to **design and implement a feedforward neural network (NN)** trained using the **back-propagation algorithm** to solve a **multi-class classification problem**.

The network can be:
- A **Fully Connected Neural Network (MLP)**,  
- A **Convolutional Neural Network (CNN)**, or  
- A **hybrid architecture** combining both.  

Training can be performed using either **supervised** or **self-supervised learning** techniques.

---

## 🧩 Description & Requirements

### 1. Neural Network Implementation
- Develop a program in any programming language that implements a feedforward NN.  
- Train it using **back-propagation**.  
- Apply it to a **multi-class classification task other than MNIST**.

---

### 2. Dataset
You may choose **one of the following datasets**:

- [`CIFAR-10`](https://www.cs.toronto.edu/~kriz/cifar.html)  
- [`CIFAR-100`](https://www.cs.toronto.edu/~kriz/cifar.html)  
- [`SVHN`](http://ufldl.stanford.edu/housenumbers/)  
- [`ImageNet100`](https://www.kaggle.com/datasets/ambityga/imagenet100)  
- [`Tiny-ImageNet`](https://huggingface.co/datasets/zh-plus/tiny-imagenet)  

Alternatively, any **multi-class classification dataset** from [Kaggle Datasets](https://www.kaggle.com/datasets) can be used.

If the dataset does not provide a predefined test set, it must be **randomly split** as follows:
- 60% for training  
- 40% for testing  
or use a **cross-validation** technique.

---

### 3. Feature Extraction
- Either use the entire input directly or select **meaningful features** (e.g., pixel intensities, average brightness across rows/columns, etc.).  
- Optionally, apply **dimensionality reduction using PCA (Principal Component Analysis)**.

---

### 4. Evaluation and Reporting
A written report must include:
- A detailed **description of the implemented algorithm and architecture**.  
- **Examples** of correctly and incorrectly classified samples.  
- **Performance metrics** (accuracy) for:
  - Training phase  
  - Testing phase  
- **Training time** and results for different:
  - Hidden layer sizes  
  - Learning parameters (e.g., learning rate, epochs, activation functions, etc.)
- **Comparative performance analysis** with:
  - **Nearest Neighbor (NN) classifier**  
  - **Nearest Class Centroid (NCC) classifier**  
- **Discussion and interpretation** of the results.  
- **Code documentation** and comments.

---

## 🧰 Tools and Frameworks
The implementation may be done in any language. However, use of **Deep Learning frameworks** is encouraged:

- [TensorFlow](https://www.tensorflow.org/)  
- [PyTorch](https://pytorch.org/)  
- [Keras](https://keras.io/)

Additionally, **self-supervised contrastive learning** can be explored (e.g., [MIFA-Lab/contrastive2021](https://github.com/MIFA-Lab/contrastive2021)).

---

## 📅 Deadlines
- **Intermediate Assignment:** *November 10, 2024*  
  Implement and compare the performance of:
  - **Nearest Neighbor classifier** (k = 1 and k = 3)  
  - **Nearest Class Centroid classifier**
  using the chosen dataset.

- **Final Assignment:** *November 24, 2024 (23:59)*  
  Late submissions are penalized by **−10% per day** (up to 5 days).  
  After all submissions, a **presentation and oral examination** will follow, including discussion of the code.

---

## 📚 References
- [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)  
- [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)  
- [ImageNet100 (Kaggle)](https://www.kaggle.com/datasets/ambityga/imagenet100)  
- [Tiny-ImageNet (HuggingFace)](https://huggingface.co/datasets/zh-plus/tiny-imagenet)  
- [Contrastive Self-Supervised Learning Example](https://github.com/MIFA-Lab/contrastive2021)  

---

**Author:** [ntoylker](https://github.com/ntoylker)
   - Περιγραφή αλγορίθμου και αρχιτεκτονικής.
   - Παράθεση **χαρακτηριστικών παραδειγμάτων ορθής και εσφαλμένης ταξινόμησης**.
   - Παρουσίαση **ποσοστών επιτυχίας** για:
     - Εκπαίδευση (training)
     - Έλεγχο (testing)
   - Σύγκριση αποτελεσμάτων για διαφορετικό:
     - αριθμό νευρώνων στο κρυφό επίπεδο  
     - παραμέτρους εκπαίδευσης  
   - Σύγκριση με τους απλούς ταξινομητές:
     - **Πλησιέστερος Γείτονας (Nearest Neighbor)**
     - **Πλησιέστερο Κέντρο Κλάσης (Nearest Class Centroid)**
   - Σχολιασμός αποτελεσμάτων και κώδικα.

---

## 🧰 Εργαλεία και Τεχνολογίες
Η υλοποίηση μπορεί να γίνει σε οποιαδήποτε γλώσσα, ωστόσο προτείνεται η χρήση Deep Learning βιβλιοθηκών:

- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [Keras](https://keras.io/)

---

## 📅 Προθεσμίες
- **Ενδιάμεση Εργασία:** 10 Νοεμβρίου 2024  
  Σύγκριση ταξινομητών Nearest Neighbor (με 1 και 3 γείτονες) και Nearest Class Centroid.
- **Τελική Υποχρεωτική Εργασία:** 24 Νοεμβρίου 2024 (ώρα 24:00)  
  Καθυστέρηση έως 5 ημέρες μειώνει τον βαθμό κατά 10% ανά ημέρα.  
  Μετά την παράδοση, ακολουθεί **παρουσίαση και προφορική εξέταση** πάνω στην εργασία και τον κώδικα.

---

## 📚 Παραπομπές
- [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)  
- [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)  
- [ImageNet100 (Kaggle)](https://www.kaggle.com/datasets/ambityga/imagenet100)  
- [Tiny-ImageNet (HuggingFace)](https://huggingface.co/datasets/zh-plus/tiny-imagenet)  
- [Contrastive Self-Supervised Learning Example](https://github.com/MIFA-Lab/contrastive2021)

---

**Συντάκτης:** [ntoylker](https://github.com/ntoylker)
