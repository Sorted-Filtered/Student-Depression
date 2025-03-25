# Predicting Depression Among Students
### Project 4
[Initial Project Proposal](https://github.com/Sorted-Filtered/Student-Depression/blob/main/Project%20proposal%20-%20project%204.docx)

[Dataset taken from Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset/data)

[Tableau Visualizations from Dataset](https://www.tableau.com)  ## Update with Tableau Public Visualization Workbook!

[Google Slides Presentation](https://docs.google.com/presentation/d/1EdIuKcOOn9DrwLD_oTe0_vwz1WLMgdN1JRT_LCTdI54/edit?usp=sharing)

## Project Description
The aim of this project is to predict depression among college students in India. To do this, our random forest model will look at a variety of factors including demographics, academic indicators, lifestyle, the academic degree pursued, and family history. The dataset comes from an anonymous survey given to college students in India. This project provides a website that predicts depression in students based on certain student parameters. Users can choose from dropdowns on the website and press a button to receive a prediction of depression status from a trained random forest model with an 85% accuracy. 

## Execution
Before running the jupyter notebook file and app, ensure you have the following installed:

-Python 3.x

-Jupyter Notebook (pip install notebook)

-Flask (pip install flask)

-Flask-CORS (pip install flask-cors)

1. Once installed, run the jupyter notebook file in the model folder named "RandomForestModel.ipynb". This will save the generated random forest model and StandardScaler to the api folder for use with the website.
2. Run the API script (api.py) found in the api folder through a terminal. Once running, you will see flask running on http://127.0.0.1:5000. Endpoints include:
```
     / : Lists available API routes
   
     /predict : Fetches model prediction output based on dropdown boxes selected on the website.
```
4. Run the app using a local web server, or open the HTML directly.
```
     Using python HTTP server: run "python -m http.server" in terminal in application's directory. This will generate the webpage on http://localhost:8000.
   
     Opening the file directly: Open HTML file in web browser, ensure that the API is running correctly to populate prediction.
```
## Features
**Website:**

-includes drop down boxes to make selections from various categories included in the dataset.

-Predict button that takes selected dropdown inputs, sends it to an API, receives output prediction, and displays it in the webpage.


**Random Forest Model:**

-Includes SQL integration, short analysis of dataset, data preprocessing (splitting and scaling), and training/testing the model.

-Includes plotted "importances" AKA weights or bias of features the model used to predict depression, a confusion matrix to see predictive error categories, and a classification report for further breakdown.

-Includes mapped Random Forest Decision Tree from the model, as well as saving trained model/scaler for use in the API.


**API:**

-Includes function that takes input from website, formats it into the correct format, and uses saved scaler/model to make a prediction based on that input which is sent back to the website.

Python modules used include Pandas, joblib, matplotlib, sklearn, Flask, Flask-CORS, and sqlite3.


## Contributors
This project was done jointly by Connor Casey, Jenna Anderson, Dylan Mavencamp, Alston C. Armah, and Lance Peterson.

## License
GPL-3.0 License
