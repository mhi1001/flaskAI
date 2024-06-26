# Diabetes Prediction Flask App
Exam project

## Table of Contents
- [Research](#research)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)


## Research
https://colab.research.google.com/drive/1CVQKCJN0vUsZ845fnMFBiDOdaKkEAtRJ#scrollTo=qVmhQp7SjGfe

## Requirements

- Python 3.6+
- pip (Python package installer)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/mhi1001/flaskAI.git
    cd flaskAI
    ```

2. **Create and activate a virtual environment:**

    On Windows:

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

    On macOS/Linux:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the trained model and scaler files:**

    Ensure you have the trained model (`diabetes_logistic_model.pkl`). Place it in the root directory of the folder.

5. **Create and populate the `label_encoder_classes` files:**

    Ensure you have the `label_encoder_diabetes_classes.npy` file. Place it in the root directory of the folder.

## Running the Application

1. **Start the Flask application:**

    ```bash
    python app.py
    ```

2. **Open your web browser and go to:**

    ```
    http://127.0.0.1:5000/
    ```

## Usage

1. Fill out the form with the required health metrics and personal information.
2. Click the "Predict" button.
3. The prediction result will be displayed on the page.


