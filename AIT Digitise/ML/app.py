# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder

# app = Flask(__name__)

# # Load dataset
# df = pd.read_csv("heart_attack_prediction_dataset.csv")

# # Drop unnecessary columns
# df1 = df.drop(['Patient ID','Income','Country','Continent','Hemisphere','Cholesterol'], axis=1)

# # Convert 'Sedentary Hours Per Day' to integer
# df1['Sedentary Hours Per Day'] = df1['Sedentary Hours Per Day'].astype(int)

# # Encode categorical variables
# le = LabelEncoder()
# df1['Sex'] = le.fit_transform(df1['Sex'])
# df1['Diet'] = le.fit_transform(df1['Diet'])

# # Split 'Blood Pressure' into 'BP1' and 'BP2'
# def split_blood_pressure(blood_pressure):
#     return pd.Series(blood_pressure.split('/', 1))

# df1[['BP1', 'BP2']] = df1['Blood Pressure'].apply(split_blood_pressure)
# df1 = df1.drop(['Blood Pressure','Triglycerides','Sedentary Hours Per Day'], axis=1)

# # Convert 'BP1' and 'BP2' to numeric
# df1['BP1'] = pd.to_numeric(df1['BP1'], errors='coerce')
# df1['BP2'] = pd.to_numeric(df1['BP2'], errors='coerce')

# # Define weights for features
# weights = {
#     'Age': 0.05,
#     'Sex': 0.05,
#     'Heart Rate': 0.3,
#     'Diabetes': 0.15,
#     'Family History': 0.1,
#     'Smoking': 0.2,
#     'Obesity': 0.1,
#     'Alcohol Consumption': 0.2,
#     'Exercise Hours Per Week': 0.05,
#     'Diet': 0.1,
#     'Previous Heart Problems': 0.3,
#     'Medication Use': 0.05,
#     'Stress Level': 0.15,
#     'BMI': 0.1,
#     'Physical Activity Days Per Week': 0.05,
#     'Sleep Hours Per Day': 0.15,
#     'Heart Attack Risk': 30,
#     'BP1': 0.2,
#     'BP2': 0.2
# }
 
# # Modify weights based on conditions
# for index, row in df1.iterrows():
#     if row['Age'] >= 45:
#         weights['Age'] = 0.2
#     if row['Sex'] == 0:
#         weights['Sex'] = 0.1
#     if row['Heart Rate'] < 60:
#         weights['Heart Rate'] = 10 + (row['Heart Rate'] - 1) * 0.02    #change
#     elif row['Heart Rate'] > 100:
#         weights['Heart Rate'] = 0.2 + (row['Heart Rate'] - 100) * 0.02
#     if row['BP1'] > 150:
#         weights['BP1'] = 0.2 + (row['BP1'] - 150) * 0.02
#     if row['BP2'] > 90:
#         weights['BP2'] = 0.2 + (row['BP2'] - 90) * 0.02

# # Calculate total weighted sum
# total_weighted_sum = df1.apply(lambda row: sum(row[col] * weights[col] for col in df1.columns), axis=1)

# # Normalize total weighted sum
# max_weighted_sum = total_weighted_sum.max()
# min_weighted_sum = total_weighted_sum.min()
# df1['percentage'] = ((total_weighted_sum - min_weighted_sum) / (max_weighted_sum - min_weighted_sum)) * 100

# # Features and target variable
# X = df1.drop(columns=['Heart Attack Risk','percentage'])
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Initialize and train Random Forest Classifier
# random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# random_forest_classifier.fit(X_scaled, df1['Heart Attack Risk'])

# # Initialize and train Random Forest Regressor
# random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# random_forest_regressor.fit(X_scaled, df1['percentage'])

# # Function to predict manually
# def predict_manually(age, sex, heart_rate, diabetes, family_history, smoking, obesity, alcohol_consumption,
#                      exercise_hours_per_week, diet, previous_heart_problems, medication_use, stress_level,
#                      bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2):
#     input_data = pd.DataFrame([[age, sex, heart_rate, diabetes, family_history, smoking, obesity, alcohol_consumption,
#                                 exercise_hours_per_week, diet, previous_heart_problems, medication_use, stress_level,
#                                 bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2]],
#                               columns=X.columns)
#     input_scaled = scaler.transform(input_data)
#     predicted_heart_attack_risk = random_forest_classifier.predict(input_scaled)[0]
#     predicted_percentage = random_forest_regressor.predict(input_scaled)[0]
#     return predicted_heart_attack_risk, predicted_percentage


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         age = int(request.form['age'])
#         sex = int(request.form['sex'])
#         heart_rate = int(request.form['heart_rate'])
#         diabetes = int(request.form['diabetes'])
#         family_history = int(request.form['family_history'])
#         smoking = int(request.form['smoking'])
#         obesity = int(request.form['obesity'])
#         alcohol_consumption = int(request.form['alcohol_consumption'])
#         exercise_hours_per_week = float(request.form['exercise_hours_per_week'])
#         diet = int(request.form['diet'])
#         previous_heart_problems = int(request.form['previous_heart_problems'])
#         medication_use = int(request.form['medication_use'])
#         stress_level = int(request.form['stress_level'])
#         bmi = float(request.form['bmi'])
#         physical_activity_days_per_week = int(request.form['physical_activity_days_per_week'])
#         sleep_hours_per_day = float(request.form['sleep_hours_per_day'])
#         bp1 = int(request.form['bp1'])
#         bp2 = int(request.form['bp2'])

#         predicted_heart_attack_risk, predicted_percentage = predict_manually(age, sex, heart_rate, diabetes,
#                                                                              family_history, smoking, obesity,
#                                                                              alcohol_consumption,
#                                                                              exercise_hours_per_week, diet,
#                                                                              previous_heart_problems, medication_use,
#                                                                              stress_level, bmi,
#                                                                              physical_activity_days_per_week,
#                                                                              sleep_hours_per_day, bp1, bp2)

#         return render_template('result.html', heart_attack_risk=predicted_heart_attack_risk,
#                                percentage=predicted_percentage)
#     else:
#         return render_template('index.html')
# from flask import Flask, render_template, request
# import mysql.connector

# app = Flask(__name__)

# # Database configuration
# db_config = {
#     'user': 'root',
#     'password': 'prajwaltp',
#     'host': 'localhost',
#     'database': 'cardiocare'
# }

# # Function to fetch user data from database
# def fetch_user_data(name):
#     try:
#         connection = mysql.connector.connect(**db_config)
#         cursor = connection.cursor(dictionary=True)
#         query = "SELECT * FROM PatientInfo WHERE name = %s"
#         cursor.execute(query, (name,))
#         user_data = cursor.fetchone()
#         return user_data
#     except mysql.connector.Error as error:
#         print("Error fetching user data:", error)
#     finally:
#         if (connection.is_connected()):
#             cursor.close()
#             connection.close()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/take', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         name = request.form['name']
#         user_data = fetch_user_data(name)
#         if user_data:
#             return render_template('index.html', user_data=user_data)
#         else:
#             return "User not found in the database"
#     else:
#         return render_template('index.html')


# if __name__ == '__main__':
#     app.run(debug=True, port=3005)



from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import mysql.connector

app = Flask(__name__)

# Database configuration
db_config = {
    'user': 'root',
    'password': 'prajwaltp',
    'host': 'localhost',
    'database': 'cardiocare'
}

# Function to fetch user data from database
def fetch_user_data(name):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM PatientInfo WHERE name = %s"
        cursor.execute(query, (name,))
        user_data = cursor.fetchone()
        return user_data
    except mysql.connector.Error as error:
        print("Error fetching user data:", error)
        return None
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# Load dataset
df = pd.read_csv("heart_attack_prediction_dataset.csv")

# Drop unnecessary columns
df1 = df.drop(['Patient ID', 'Income', 'Country', 'Continent', 'Hemisphere', 'Cholesterol'], axis=1)

# Convert 'Sedentary Hours Per Day' to integer
df1['Sedentary Hours Per Day'] = df1['Sedentary Hours Per Day'].astype(int)

# Encode categorical variables
le = LabelEncoder()
df1['Sex'] = le.fit_transform(df1['Sex'])
df1['Diet'] = le.fit_transform(df1['Diet'])

# Split 'Blood Pressure' into 'BP1' and 'BP2'
def split_blood_pressure(blood_pressure):
    return pd.Series(blood_pressure.split('/', 1))

df1[['BP1', 'BP2']] = df1['Blood Pressure'].apply(split_blood_pressure)
df1 = df1.drop(['Blood Pressure', 'Triglycerides', 'Sedentary Hours Per Day'], axis=1)

# Convert 'BP1' and 'BP2' to numeric
df1['BP1'] = pd.to_numeric(df1['BP1'], errors='coerce')
df1['BP2'] = pd.to_numeric(df1['BP2'], errors='coerce')

# Define weights for features
weights = {
    'Age': 0.05,
    'Sex': 0.05,
    'Heart Rate': 0.3,
    'Diabetes': 0.15,
    'Family History': 0.1,
    'Smoking': 0.2,
    'Obesity': 0.1,
    'Alcohol Consumption': 0.2,
    'Exercise Hours Per Week': 0.05,
    'Diet': 0.1,
    'Previous Heart Problems': 0.3,
    'Medication Use': 0.05,
    'Stress Level': 0.15,
    'BMI': 0.1,
    'Physical Activity Days Per Week': 0.05,
    'Sleep Hours Per Day': 0.15,
    'Heart Attack Risk': 30,
    'BP1': 0.2,
    'BP2': 0.2
}

# Modify weights based on conditions
for index, row in df1.iterrows():
    if row['Age'] >= 45:
        weights['Age'] = 0.2
    if row['Sex'] == 0:
        weights['Sex'] = 0.1
    if row['Heart Rate'] < 60:
        weights['Heart Rate'] = 10 + (row['Heart Rate'] - 1) * 0.02
    elif row['Heart Rate'] > 100:
        weights['Heart Rate'] = 0.2 + (row['Heart Rate'] - 100) * 0.02
    if row['BP1'] > 150:
        weights['BP1'] = 0.2 + (row['BP1'] - 150) * 0.02
    if row['BP2'] > 90:
        weights['BP2'] = 0.2 + (row['BP2'] - 90) * 0.02

# Calculate total weighted sum
total_weighted_sum = df1.apply(lambda row: sum(row[col] * weights[col] for col in df1.columns), axis=1)

# Normalize total weighted sum
max_weighted_sum = total_weighted_sum.max()
min_weighted_sum = total_weighted_sum.min()
df1['percentage'] = ((total_weighted_sum - min_weighted_sum) / (max_weighted_sum - min_weighted_sum)) * 100

# Features and target variable
X = df1.drop(columns=['Heart Attack Risk', 'percentage'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train Random Forest Classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_scaled, df1['Heart Attack Risk'])

# Initialize and train Random Forest Regressor
random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_regressor.fit(X_scaled, df1['percentage'])

# Function to predict manually
def predict_manually(age, sex, heart_rate, diabetes, family_history, smoking, obesity, alcohol_consumption,
                     exercise_hours_per_week, diet, previous_heart_problems, medication_use, stress_level,
                     bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2):
    input_data = pd.DataFrame([[age, sex, heart_rate, diabetes, family_history, smoking, obesity, alcohol_consumption,
                                exercise_hours_per_week, diet, previous_heart_problems, medication_use, stress_level,
                                bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2]],
                              columns=X.columns)
    input_scaled = scaler.transform(input_data)
    predicted_heart_attack_risk = random_forest_classifier.predict(input_scaled)[0]
    predicted_percentage = random_forest_regressor.predict(input_scaled)[0]
    return predicted_heart_attack_risk, predicted_percentage

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        user_data = fetch_user_data(name)
        if user_data:
            # Extract user data
            age = int(user_data['age'])
            diabetes = int(user_data['diabetes'])
            family_history = int(user_data['famhistory'])
            smoking = int(user_data['smoking'])
            obesity = int(user_data['obesity'])
            alcohol_consumption = int(user_data['alcohol'])
            exercise_hours_per_week = int(user_data['exercise'])
            
            # Handle diet attribute
            diet_mapping = {'Poor': 0, 'Average': 1, 'Good': 2}
            diet = diet_mapping.get(user_data['diet'], -1)  # Default to -1 if diet is not found in the mapping
            if diet == -1:
                return "Invalid diet value"
            
            previous_heart_problems = int(user_data['prevheart'])
            medication_use = int(user_data['meduse'])
            bmi = float(user_data['bmi'])
            physical_activity_days_per_week = int(user_data['phyactivity'])  # Assuming encoded value
            sleep_hours_per_day = float(user_data['sleep'])
            bp1 = float(user_data['bp1'])
            bp2 = float(user_data['bp2'])
            # Fetch heart rate and stress level from form input
            heart_rate = int(request.form['heart_rate'])
            stress_level = int(request.form['stress_level'])
            
            # Handle sex attribute
            sex_mapping = {'male': 0, 'female': 1}
            sex = sex_mapping.get(request.form['sex'], -1)  # Default to -1 if sex is not found in the mapping
            if sex == -1:
                return "Invalid sex value"
            
            # bp_values = user_data['bp1'].split('/')
            # if len(bp_values) != 2:
            #     return "Invalid blood pressure format"
            # bp1, bp2 = map(int, bp_values)            

            # Print the extracted data for debugging
            print("Extracted User Data:")
            print("Age:", age)
            print("Sex:", sex)
            # Print other extracted data similarly

            # Predict heart attack risk using the extracted data and form input
            predicted_heart_attack_risk, predicted_percentage = predict_manually(age, sex, heart_rate, diabetes,
                                                                                 family_history, smoking, obesity,
                                                                                 alcohol_consumption,
                                                                                 exercise_hours_per_week, diet,
                                                                                 previous_heart_problems, medication_use,
                                                                                 stress_level, bmi,
                                                                                 physical_activity_days_per_week,
                                                                                 sleep_hours_per_day, bp1, bp2)

            # Print the predicted values for debugging
            print("Predicted Values:")
            print("Heart Attack Risk:", predicted_heart_attack_risk)
            print("Percentage:", predicted_percentage)

            return render_template('result.html', heart_attack_risk=predicted_heart_attack_risk,
                                   percentage=predicted_percentage)
        else:
            return "User not found in the database"
    else:
        return render_template('index.html')


@app.route('/take', methods=['POST'])
def take():
    if request.method == 'POST':
        name = request.form['name']
        user_data = fetch_user_data(name)
        if user_data:
            return render_template('index.html', user_data=user_data)
        else:
            return "User not found in the database"
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3005)
