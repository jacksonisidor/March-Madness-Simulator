from flask import Flask, render_template

# Initialize the Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
