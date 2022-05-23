# Solving-Sudoku-Using-DL
# Solving Sudoku Using Image Detection, Digit Recognition and Deep Learning

Solving the sudoku that is printed in the daily newspaper is something that we have all grown up doing. However, the problem with this is that we cannot confirm the correctness of the solution until the next day since the solution is printed in the following day’s newspaper. From the reference links provided to us by the professor, we have chosen the topic of Solving Sudoku using Deep Learning. Multiple machine learning algorithms already exist that solve the sudoku, but development is still in progress in the field of taking sudoku image as input. For a layman, entering the sudoku in array format would be a tedious task as compared to clicking a picture of the sudoku and uploading it. We thus propose a technique where in only the sudoku image will be required as the input and the algorithm will output the solved sudoku.

We have used two deep learning models that have been trained and are being used in two different phases of our project. The main architecture of implementation follows the below depicted flow.

 
![alt text](https://github.com/desainikita/Solving-Sudoku-Using-DL/blob/main/Screen%20Shot%202022-05-22%20at%209.43.36%20PM.png?raw=true)

# Getting Started

Folder Structure :

| File name | Description & purpose |
| :---: | :---: |
| main.ipynb | Main .ipynb file ( Starting point of execution) |
| model.py | Contains deep learning model definitions |
| process_image.py | Contains functions for image pre-processing ( reframing, warping, croppping, splitting , contouring ) |
| load_data | Loads data for training of both models |
| sudoku.py | Solving sudoku puzzle using Deep neural net model |
| dataset/Digits | Must contain the dataset prepared from the official Chars74K dataset |
| cropnum/ | Intermediate folder created in code for saving cropped puzzle images |
| models/ | Pre-trained models saved after training for digit recognition and for solving the puzzle|
| pickled_files/ | Saved model training history for analysis later |
| sudoku_samples/ | Some sample sudoku puzzle images |
| utility/ | Utility functions for pickling and plotting |


How to use :

1. Creating the dataset
    We have used a subset of the Chars74K dataset containing images of only digits in different fonts.
    ( Reference for the Char74k dataset : http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/#download )
    Steps for dataset creation:
    1) Download "Digits" Folder from "https://drive.google.com/drive/folders/1A09-g7j5c0XV9R-IUUdZl9wqqXdsSTDI?usp=sharing"
    2) Copy the "Digits" Folder and paste inside the "dataset" folder in this project.
    3) Download sudoku dataset from " https://www.kaggle.com/datasets/rohanrao/sudoku"
    4) Copy and paste sudoku.csv into the "dataset" folder in this project.
  
2. Run
    This code has been implemented in Google Colab. After opening Google colab, follow below steps
    1) Open main.ipynb
    2) Uncomment the first two blocks to clone the repository
    ```
    git clone "https://github.com/desainikita/Solving-Sudoku-Using-DL"
    
    import os
    os.chdir("/content/Solving-Sudoku-Using-DL")

    ```
3. Run all cells



