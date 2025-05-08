
# Disaster Classification

This repository contains a Convolutional Neural Network (CNN) model for classifying natural disaster images into four categories: **Cyclones**, **Earthquakes**, **Floods**, and **Wildfires**. The model achieves high accuracy with near-perfect AUC scores across all classes.

---

## Project Overview

This deep learning project aims to automatically identify natural disasters from images. It has applications in:

- Emergency response
- Disaster management
- Early warning systems

---

## Dataset

The model is trained on the **Natural Disaster Image Dataset**, which includes labeled images of:

- Cyclones 
- Earthquakes 
- Floods 
- Wildfires

---

## Model Architecture

The custom CNN architecture includes:

- Multiple convolutional layers with **batch normalization**
- **Max pooling** for downsampling
- **Global average pooling** before dense layers
- **Dense layers** with **dropout** for regularization
- **L2 regularization** to prevent overfitting

---

## Performance Metrics

| Metric                  | Score |
|-------------------------|-------|
| **Test Accuracy**       | ~97%  |
| **F1 Score (macro avg)**| ~0.97 |
| **AUC (Cyclone)**       | 1.00  |
| **AUC (Earthquake)**    | 0.98  |
| **AUC (Flood)**         | 0.98  |
| **AUC (Wildfire)**      | 0.98  |

---

## Interactive Demo Application

The project includes a **Streamlit** web app that allows users to:

- Upload images for classification
- View predictions with confidence scores
- Access disaster-specific **evacuation guidelines**
- Plan **evacuation routes** via Google Maps integration

---

## 🛠Setup and Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.7+
- pip (Python package manager)
- git

### Installation Steps

1. **Clone this repository**

```bash
git clone https://github.com/yourusername/natural-disaster-classification.git
cd natural-disaster-classification
```

2. **Install required dependencies**

Make sure to create a virtual environment if desired:

```bash
# Optional: Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

<details>
<summary><code>requirements.txt</code> contents</summary>

```txt
tensorflow>=2.8.0
streamlit
pillow
matplotlib
scikit-learn
numpy
pandas
opencv-python
googlemaps
```
</details>

---

### Downloading the Pre-trained Model

You have two options:

#### Option 1: Download from release page

- Go to the [Releases](https://github.com/yourusername/natural-disaster-classification/releases) section of this repository.
- Download the file `cnn_model_finetuned.h5`
- Place it in the root project directory.

#### Option 2: Train your own model

If you'd like to train from scratch:

```bash
jupyter notebook train_model.ipynb
```

The trained model will be saved automatically as:

```bash
cnn_model_finetuned.h5
```

---

### Running the Streamlit App

After installing dependencies and adding the model:

```bash
streamlit run app.py
```

The application will open in your browser at:

[http://localhost:8501](http://localhost:8501)

You can now upload an image and receive real-time disaster classification along with evacuation routes and safety guidelines!

---

## Training Your Own Model

To train the model from scratch:

1. Download the **Natural Disaster Image Dataset**
2. Run the Jupyter notebook:

```bash
jupyter notebook train_model.ipynb
```

3. The trained model will be saved as `cnn_model_finetuned.h5`

---

## Usage Guide

### Image Classification

- Upload an image of a potential disaster
- View the model's predicted class and confidence score
- Access safety guidelines tailored to the disaster type

### Evacuation Planning

- Enter your current location and safe destination
- Provide your Google Maps API key
- Get a direct evacuation route via Google Maps

---

## Future Work

- Mobile app development for offline usage
- Integration with real-time weather and seismic data
- Multi-language support for broader access
- Expanded disaster categories and sub-classes

---

## License

This project is licensed under the **Apache License 2.0**.

---

## Acknowledgments

- Dataset providers
- TensorFlow and Keras teams
- Streamlit for their interactive app framework

---

Feel free to contribute or report issues to improve this project!
