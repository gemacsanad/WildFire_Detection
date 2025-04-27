# Wildfire Detection Using Satellite Images
## Gema Csanád

This project aims to develop a deep learning model for detecting wildfires in satellite images. 
The dataset, sourced from Kaggle, consists of images labeled as **wildfire** and **no-wildfire**. 
Data augmentation techniques such as rotation, flipping, resizing, and color adjustments are applied to enhance the model's generalization. 
The project will explore challenges such as distinguishing smoke from clouds and handling images where fire is not visibly apparent.
The outcome of this project will be a trained model capable of detecting wildfires from satellite imagery.


## Telepítés

A projekt futtatásához a következő lépések szükségesek:

1.  **Klónozd a repót a gépedre:**
    git clone [https://github.com/gemacsanad/WildFire_Detection/](https://github.com/gemacsanad/WildFire_Detection/)
    cd Wildfire_Detection
    

2.  **Hozd létre egy virtuális környezetet (ajánlott):**
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate  # Windows

3.  **Telepítsd a szükséges függőségeket a `requirements.txt` fájlból:**
    pip install -r requirements.txt

## Futtatás

A projekt futtatásához kövesd az alábbi lépéseket:

1.  **Dataset letötlése:**
    Futtasd a `milestone1.py` szkriptet.
    ```bash
    python milestone1.py
    ```

2.  **Modell tanítása:**
    Az adathalmaz letöltése után futtasd a `model.py` szkriptet a modell betanításához.
    ```bash
    python model.py
    ```
