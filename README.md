# CSCI-635-1-Group-5

Project Name:
    Forest Cover Classification From Local Conditions

Abstract:
	Our project was to classify the type of forest cover given local conditions gathered by researchers from the Roosevelt National Forest in Colorado. The provided dataset didn't contain any missing values, or unlabeled data, however the represenation of classes was poor with class 0 and 1 representing over 80% of the data. To deal with this inequality in representation we utilized oversampling, undersampling, and balancing class weights to improve model performance. The models that were created in this project include: K Nearest Neighbors, Logistic Regression, Neural Network, and an Ensemble model. 

Members:
    - Sebatian Banasik:
        - Logistic Regression
        - AdaBooost of Decisison Trees
    - Alexander Jermyn:
        - K Nearest Neighbors
        - Multi Layer Perceptron
        - Preprocessing scripts

How To Run The Project:
    1) Install all the requirements with the command "pip install requirements.txt".
    2) All notbook outputs are saved so running the notebook is not required.
        - Note that running the notebooks can result in a very long runtime.
        - When running a notebook run each cell in order.
