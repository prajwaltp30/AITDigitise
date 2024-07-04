from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os
import numpy as np
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)


db_config = {
    'user': 'root',
    'password': 'prajwaltp',
    'host': 'localhost',
    'database': 'cardiocare'
}

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/start', methods=['POST'])
# def start_script():
#     if request.method == 'POST':
#         subprocess.Popen(['python', 'merge.py'])
#         return 'Script started successfully'

#     return render_template('index.html')

@app.route('/results', methods=['GET', 'POST'])  # Allow both GET and POST requests
def show_results():
    if request.method == 'POST':
        # Handle form submission if needed
        pass



    # Determine if an additional graph exists
    additional_graph_exists = os.path.exists("final_graph.png")

    return render_template('results.html', additional_graph_exists=additional_graph_exists)

    # Read BPM data from the file
    bpm_data = []
    with open("bpm_values.txt", "r") as file:
        for line in file:
            bpm = float(line.strip())
            bpm_data.append(bpm)

    # Calculate average BPM between 5 to 20 seconds
    average_bpm = np.mean(bpm_data[5:])

    return render_template('index.html', final_readings=final_readings, additional_graph_exists=additional_graph_exists, average_bpm=average_bpm)

@app.route('/submit_health_info', methods=['POST'])
def submit_health_info():
    if request.method == 'POST':
        # Retrieve form data
        name = request.form['name']
        age = request.form['age']
        diabetes = request.form['diabetes']
        famhistory = request.form['famhistory']
        smoking = request.form['smoking']
        obesity = request.form['obesity']
        alcohol = request.form['alcohol']
        exercise = request.form['exercise']
        diet = request.form['diet']
        prevheart = request.form['prevheart']
        meduse = request.form['meduse']
        bmi = request.form['bmi']
        phyactivity = request.form['phyactivity']
        sleep = request.form['sleep']
        bp1 = request.form['bp1']
        bp2 = request.form['bp2']

        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO PatientInfo (name, age, diabetes, famhistory, smoking, obesity, alcohol, exercise, diet, prevheart, meduse, bmi, phyactivity, sleep, bp1, bp2) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                           (name, age, diabetes, famhistory, smoking, obesity, alcohol, exercise, diet, prevheart, meduse, bmi, phyactivity, sleep, bp1, bp2))
            conn.commit()
            cursor.close()
            conn.close()
            return redirect(url_for('index'))  # Redirect to index page after form submission
        except mysql.connector.Error as err:
            return f"Error: {err}"

    return render_template('info.html')

@app.route('/user_info')
def user_info():
    return render_template('info.html')
if __name__ == '__main__':
    app.run(debug=True)
