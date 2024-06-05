# Diabetes Prediction Flask App
Exam project

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

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

    Ensure you have the trained model (`diabetes_logistic_modelMINE.pkl`) and scaler (`scaler.pkl`) files. Place these files in the root directory of the folder.

5. **Create and populate the `label_encoder_classes` files:**

    Ensure you have the `label_encoder_gender_classes.npy`, `label_encoder_diabetes_classes.npy`, `label_encoder_bmi_classes.npy`, and `label_encoder_bp_classes.npy` files. Place these files in the root directory of the folder.

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


