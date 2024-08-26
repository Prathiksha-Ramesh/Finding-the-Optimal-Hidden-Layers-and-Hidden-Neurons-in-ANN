# Finding the Optimal Hidden Layers and Hidden Neurons in ANN

## Overview

This project aims to determine the optimal configuration of hidden layers and neurons for an Artificial Neural Network (ANN) in the context of customer churn prediction. By experimenting with different ANN architectures, the goal is to identify the most effective model for predicting customer churn. The project leverages TensorFlow, Scikit-learn, and other Python libraries for model building, evaluation, and deployment.

## Project Structure

- **notebook.ipynb**: A Jupyter notebook that contains the detailed exploration of various ANN architectures, including data preprocessing, model training, hyperparameter tuning, and performance evaluation.
- **Churn_Modelling.csv**: The dataset used for training and evaluating the ANN models. It includes customer attributes like geography, gender, age, tenure, and balance.
- **label_encoder_gender.pkl**: A pickle file storing the label encoder for the 'Gender' feature used in the model.
- **onehot_encoder_geo.pkl**: A pickle file storing the one-hot encoder for the 'Geography' feature used in the model.
- **scaler.pkl**: A pickle file storing the scaler object used to standardize the numeric features in the dataset.
- **requirements.txt**: A text file listing all the required Python packages to run the project.
- **LICENSE**: The license file for the project.
- **.gitignore**: Specifies files and directories that should be ignored by git.

## Installation

To run this project locally, follow these steps:

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/optimal-ann-layers.git
    cd optimal-ann-layers
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Jupyter notebook:
    ```bash
    jupyter notebook notebook.ipynb
    ```

## Usage

1. **Data Preprocessing**:
    - The dataset undergoes various preprocessing steps, including label encoding, one-hot encoding, and feature scaling. The encoders and scaler are saved as `.pkl` files for reuse in the model.

2. **Model Training**:
    - Multiple ANN architectures are trained with different numbers of hidden layers and neurons. The training process includes hyperparameter tuning to find the optimal model configuration.

3. **Model Evaluation**:
    - Each model is evaluated based on accuracy, precision, recall, and F1-score. The results are visualized using Matplotlib to compare the performance of different architectures.

4. **Deployment**:
    - The final model can be deployed using Streamlit, providing an interactive interface for predicting churn based on new customer data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have ideas for improving the project or want to add new features, please fork the repository and create a pull request.

## Acknowledgments

- This project utilizes [Scikit-learn](https://scikit-learn.org/stable/) for machine learning and data preprocessing.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) are used for building and training the ANN models.
- [Streamlit](https://streamlit.io/) is used to deploy the model as a web application for easy accessibility.
