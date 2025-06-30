# Cats-vs-dogs-classification


## 📦 Requirements
Install dependencies using pip:

pip install numpy matplotlib scikit-image scikit-learn

Or use the provided `requirements.txt`:

pip install -r requirements.txt




## 🖼️ Dataset
Ensure your directory is structured like this:

training_set/
├── cats/
│   └── cat.1.jpg, cat.2.jpg, ..., cat.2000.jpg
├── dogs/
│   └── dog.1.jpg, dog.2.jpg, ..., dog.2000.jpg

test_set/
├── cats/
│   └── cat.4001.jpg, ..., cat.5000.jpg
├── dogs/
│   └── dog.4001.jpg, ..., dog.5000.jpg


## ▶️ How to Run
Run the classifier script:

python cats_dogs_classifier.py

This will:
- Preprocess and grayscale all images
- Train and evaluate KNN and Decision Tree models
- Perform parameter optimization using GridSearch and RandomizedSearch
- Show predictions on sample test images



## 🔍 Main Components
- **Image Preprocessing**: Resize to (200x200), convert to grayscale
- **KNN Classifier**: Baseline classifier with distance metric tuning
- **Decision Tree**: Alternative classifier using Gini & Entropy
- **Hyperparameter Search**: RandomizedSearchCV and GridSearchCV
- **Prediction Function**: Custom k-NN written from scratch for manual predictions



## 📊 Results
- **KNN Accuracy (Optimized)**: ~58%
- **Decision Tree Accuracy**: ~55%

⚠️ These results highlight the limitations of classical ML for image classification — consider switching to CNNs for better performance!



## 🧠 Next Steps (Suggestions)
- Implement Convolutional Neural Networks (CNNs)
- Use pre-trained models (VGG16, ResNet, MobileNet)
- Add data augmentation and normalization






**Author**: IRGUI Ilyas

> "Traditional ML can teach you the fundamentals — but deep learning brings the firepower."
