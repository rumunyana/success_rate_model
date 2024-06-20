# Predicting Student Success Rate Using Machine Learning

## Project Overview
In this project, we aim to develop a predictive model that can identify students at risk of dropping out or underperforming academically, based on their demographic, socio-economic, and academic performance data. By leveraging machine learning techniques, specifically deep neural networks, we can uncover hidden patterns and make accurate predictions to enable timely interventions and support strategies.

## Problem Statement

The problem of student dropout and underperformance is a significant challenge in the education sector, with far-reaching consequences for individuals, families, and society. Identifying students at risk of dropping out or underperforming is crucial for implementing timely interventions and support strategies to improve student outcomes and reduce dropout rates. However, predicting student success is a complex and multifaceted problem that requires analyzing a wide range of factors, including demographic information, socio-economic status, and academic performance data.
nderstanding the multifaceted factors influencing student outcomes is crucial for designing effective interventions and support systems. Traditional approaches often fail to capture the complex interactions between various variables.


## Data
The model is trained on the following datasets:

- Socio-Economic Factors Dataset: This dataset encompasses demographic data, socio-economic indicators, and regional economic factors such as family income, parental education, financial aid, unemployment rate, inflation rate, and GDP.
- Academic Performance Dataset: This dataset includes students' academic records, such as grades, course enrollment, attendance, and other relevant academic data.

The data was collected from partner institutions and reliable economic data sources, ensuring compliance with data privacy regulations and ethical guidelines.


## Model Architecture
The core of this project is a deep neural network model designed to predict student dropout and academic success. The model architecture is as follows:

- Input Layer: Accepts various features related to student demographics, socio-economic factors, and academic performance.
- Hidden Layers: Multiple hidden layers capture complex patterns and relationships in the data.
- Output Layer: Using a softmax activation function, the output layer consists of two neurons representing the probability of student dropout and academic success.
- Regularization: Techniques such as L1/L2 regularization and dropout are applied to prevent overfitting and improve model generalization.
- Optimization: The model is trained using an appropriate optimization algorithm, such as Adam or RMSprop, to minimize the loss function and improve predictive accuracy.


## Techniques Used
To enhance the model's performance and accuracy, several techniques were employed:

- Data Preprocessing: The data was thoroughly cleaned, handling missing values and inconsistencies, and normalized to ensure optimal model training.
- Feature Engineering: Relevant features were carefully selected and engineered to improve the model's predictive power.
- Cross-Validation: Cross-validation techniques were used to evaluate the model's performance and prevent overfitting.
- Hyperparameter Tuning: Hyperparameters, such as learning rate, batch size, and number of hidden layers, were fine-tuned to optimize the model's performance.
- Ensemble Methods: Different ensemble techniques, such as bagging and boosting, were explored to combine multiple models and improve overall accuracy.


## Model Evaluation
The model's performance was evaluated using various metrics, including accuracy, precision, recall, and F1-score. The model achieved promising results, with an accuracy of 82%, a precision of 83%, a recall of 83%.


## Model Choice
The model with the highest accuracy was selected for deployment. The model was trained using the following hyperparameters: learning rate = 0.01, batch size = 32, and number of hidden layers = 3. The model achieved an accuracy of 82% on the test dataset and was deemed suitable for deployment.

## Conclusion
The model achieved an accuracy of 82% on the test dataset and was deemed suitable for deployment. This demonstrates the effectiveness of deep learning techniques in predicting student success. By identifying students at risk of dropping out or underperforming, educational institutions can implement timely interventions and support strategies to improve student outcomes and reduce dropout rates. The model's predictive power can help educators and policymakers make informed decisions and allocate resources effectively to support students' academic success.


## Future Work 
In the future, we plan to explore other machine learning techniques, such as support vector machines (SVM) and gradient boosting, to improve the model's accuracy and generalization. Additionally, we aim to incorporate more diverse and comprehensive datasets to capture a wider range of factors influencing student outcomes. By leveraging advanced machine learning algorithms and data sources, we can develop more robust predictive models to address the complex problem of student success and dropout prediction.

## Instructions for Running the Notebook and Loading the Saved Models

Prerequisites

Python 3.7+: Ensure that you have Python 3.7 or later installed on your system.

Jupyter Notebook: Install Jupyter Notebook to run the provided .ipynb file.

Required Libraries: Install the necessary Python libraries. You can do this by running:

pip install -r requirements.txt

## Running the Notebook

Clone the Repository:

git clone <repository-url>
cd <repository-directory>

Start Jupyter Notebook:

jupyter notebook

Open the Notebook: In the Jupyter Notebook interface, navigate to and open the student_success_prediction.ipynb file.

Run the Cells: Execute the cells in the notebook sequentially to preprocess the data, train the model, and evaluate its performance.

## Loading the Saved Models

Model Files: Ensure you have the saved model files (model.h5 or similar) in the appropriate directory.

Load the Model: In the notebook, use the following code snippet to load the saved model:

from keras.models import load_model

# Load the saved model
model = load_model('path/to/your/saved_model.h5')

# Example of using the model for prediction
predictions = model.predict(your_input_data)


Prediction: Use the loaded model to make predictions on new data as demonstrated in the notebook.


## References
- UNESCO Institute for Statistics. (2021). Education in Africa. Retrieved from
http://uis.unesco.org/en/topic/education-africa
- World Bank. (2020). The Impact of COVID-19 on Education in Africa and the Way
Forward. Retrieved from
https://www.worldbank.org/en/topic/education/publication/the-impact-of-covid-19-on-ed
ucation-in-africa-and-the-way-forward
- Kaggle. (2023). Predict Students' Dropout and Academic Success. Retrieved from Kaggle
Dataset
- Kaggle. (2023).Predict students' dropout and academic success
- Investigating the Impact of Social and Economic Factors Retrieved from
https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-ret
ention
