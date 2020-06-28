# Early Failure Prevention Final Exam Predictor
This project was created during my last year as Teaching Assistant

## How to use these model
1. pip -m venv venv_name
2. source venv_name/bin/activate
3. pip install -r requirements.txt
4. python main.py

## Steps
1. Cleaning Data
2. Feature Engineering
3. Choosing Model
4. Training Model
5. Evaluating Model
6. Testing and Serve the Model

## Cleaning the Data
At first there was 10 Weeks worth of student scores, and each week contains around 3-4 questions to be judged.
to clean the data, we make sure to delete all scores where the student were absent, caught cheating, etc.

## Feature Engineering
After we clean the data, directly we use all the score to predict the Final Exam Status, and the results were not good,
from there we do some Feature Engineering, from choosing which week have the highest correlation between the weekly score and Final Exam Status.
![Correlation Image](https://github.com/ad17-2/early-failure-prevention-final-exam-predictor/blob/master/correlation.jpg)

## Choosing Model
After we found our Feature, we choose K-NN to do the job, because it was simple and at that time, i just started learning.

## Evaluating Model
We evaluate the model using Cross Validation Technique with these picture as the result
![Evaluation Result](https://github.com/ad17-2/early-failure-prevention-final-exam-predictor/blob/master/evaluation.png)

## Testing and Serve the Model
After we do evaluation, we test and serve the model using Flask, where user can upload Excel containing student score and the web will show the result,
which student will pass the Final Exam or needs to be monitored.
