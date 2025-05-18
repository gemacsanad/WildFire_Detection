# Wildfire Detection Using Satellite Images
## Gema Csanád

This project aims to develop a deep learning model for detecting wildfires in satellite images. 
The dataset, sourced from Kaggle, consists of images labeled as **wildfire** and **no-wildfire**. 
The outcome of this project will be a trained model capable of detecting wildfires from satellite imagery.

## File structure

data_loading.ipynb: downloads the dataset from kaggle, preprocess the images, save the data to .npy files
small_model.ipynb: Code for First, really small model
small_model.h5: First, really small model
medium_model.ipynb: Code for medium sized model
small_model.h5: Medium model
small_model.ipynb: Code for the final model
small_model.h5: Final model
Documentation.pdf: Written report of the project
GUI.py: GUI for the project
requirements.txt: List of all packages with versions
presentation folder: conations the presentations for the project



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
    Run the `GUI.py` script.
    ```bash
    python GUI.py
    ```



