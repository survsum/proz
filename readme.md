
This is a computer vision project that utilizes machine learning techniques to process and classify images. The project includes scripts for data collection, model training, inference, and a web application interface.

## Features

* **Data Collection**: Scripts to gather and preprocess image data for training.
* **Model Training**: Tools to train a classifier on the collected dataset.
* **Inference**: Functions to perform image classification using the trained model.
* **Web Application**: A Streamlit-based interface for user interaction with the model.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/survsum/proz.git
   cd proz
   ```



2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```



## Usage

* **Data Collection**:

```bash
  python collect_imgs.py
```



* **Create Dataset**:

```bash
  python create_dataset.py
```



* **Train Classifier**:

```bash
  python train_classifier.py
```



* **Run Inference**:

```bash
  python inference_classifier.py
```



* **Launch Web Application**:

```bash
  streamlit run streamlit_app.py
```



## Project Structure

```
├── app.py
├── collect_imgs.py
├── create_dataset.py
├── data/
├── data.pickle
├── inference_classifier.py
├── model.p
├── requirements.txt
├── static/
├── streamlit_app.py
├── templates/
├── train_classifier.py
├── web_app.py
```



## License

This project is licensed under the MIT License.

