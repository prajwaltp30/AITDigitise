# from flask import Flask, request, render_template, redirect, url_for, session
# import mysql.connector
# from mysql.connector import Error

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Needed for session management

# # Database connection configuration
# db_config = {
#     'user': 'root',
#     'password': 'prajwaltp',
#     'host': 'localhost',
#     'database': 'cardiocare'
# }

# @app.route('/')
# def signup_form():
#     return render_template('signup.html')

# @app.route('/submit_signup', methods=['POST'])
# def submit_signup():
#     name = request.form['name']
#     phone_number = request.form['phonenumber']
#     email = request.form['email']
#     password = request.form['password']

#     try:
#         conn = mysql.connector.connect(**db_config)
#         cursor = conn.cursor()
#         cursor.execute("INSERT INTO UserCredentials (name, phone_number, email, password) VALUES (%s, %s, %s, %s)",
#                        (name, phone_number, email, password))
#         conn.commit()
#         cursor.close()
#         conn.close()
#         return redirect(url_for('login_form'))
#     except mysql.connector.Error as err:
#         return f"Error: {err}"

# @app.route('/login', methods=['GET', 'POST'])
# def login_form():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']

#         try:
#             conn = mysql.connector.connect(**db_config)
#             cursor = conn.cursor()
#             cursor.execute("SELECT * FROM UserCredentials WHERE email = %s AND password = %s", (email, password))
#             user = cursor.fetchone()
#             cursor.close()
#             conn.close()

#             if user:
#                 session['user'] = user[0]  # Assuming 'user' is the first column in your UserCredentials table
#                 return "Login Successful"  # You can redirect to a dashboard or another page
#             else:
#                 return "Invalid email or password"
#         except mysql.connector.Error as err:
#             return f"Error: {err}"

#     return render_template('login.html')

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)


from flask import Flask, request, render_template, redirect, url_for, session
import mysql.connector
from mysql.connector import Error
import os
import subprocess

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# Database connection configuration
db_config = {
    'user': 'root',
    'password': 'prajwaltp',
    'host': 'localhost',
    'database': 'cardiocare'
}

# Ensure the app can locate templates in both app/templates and website/templates
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

@app.route('/')
def signup_form():
    return render_template('signup.html')

@app.route('/submit_signup', methods=['POST'])
def submit_signup():
    name = request.form['name']
    phone_number = request.form['phonenumber']
    email = request.form['email']
    password = request.form['password']

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO UserCredentials (name, phone_number, email, password) VALUES (%s, %s, %s, %s)",
                       (name, phone_number, email, password))
        conn.commit()
        cursor.close()
        conn.close()
        return redirect(url_for('login_form'))
    except mysql.connector.Error as err:
        return f"Error: {err}"
@app.route('/login', methods=['GET', 'POST'])
def login_form():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM UserCredentials WHERE email = %s AND password = %s", (email, password))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user:
                session['user'] = user[0]  # Assuming 'user' is the first column in your UserCredentials table
                return redirect(url_for('index'))
            else:
                return "Invalid email or password"
        except mysql.connector.Error as err:
            return f"Error: {err}"

    return render_template('login.html')

# User info route
@app.route('/user_info')
def user_info():
    user_id = session.get('user')
    if user_id:
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM UserCredentials WHERE id = %s", (user_id,))
            user_info = cursor.fetchone()
            cursor.close()
            conn.close()

            if user_info:
                return render_template('info.html', user_info=user_info)
            else:
                return "User not found"
        except mysql.connector.Error as err:
            return f"Error: {err}"
    else:
        return redirect(url_for('login_form'))

@app.route('/index')
def index():
    # Render the 'index.html' template
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def start_script():
    if request.method == 'POST':
        subprocess.Popen(['python', 'merge.py'])
        return 'Script started successfully'

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)

