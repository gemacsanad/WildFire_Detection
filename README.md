# Wildfire Detection Using Satellite Images
## Gema Csanád

This project aims to develop a deep learning model for detecting wildfires in satellite images. 
The dataset, sourced from Kaggle, consists of images labeled as **wildfire** and **no-wildfire**. 
Data augmentation techniques such as rotation, flipping, resizing, and color adjustments are applied to enhance the model's generalization. 
The project will explore challenges such as distinguishing smoke from clouds and handling images where fire is not visibly apparent.
The outcome of this project will be a trained model capable of detecting wildfires from satellite imagery.


## Installation

To run the project, follow these steps:

1.  **Clone the repository to your machine:**
    ```bash
    git clone https://github.com/gemacsanad/WildFire_Detection/
    cd WildFire_Detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate  # Windows
    ```

3.  **Install the required dependencies from the `requirements.txt` file:**
    ```bash
    pip install -r requirements.txt
    ```

## Running

To run the project, follow these steps:

1.  **Download the dataset:**  
    Run the `milestone1.py` script.
    ```bash
    python milestone1.py
    ```

2.  **Train the model:**  
    After downloading the dataset, run the `model.py` script to train the model.
    ```bash
    python model.py
    ```

