
from flask import Flask, render_template_string, request
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("spam_model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SMS Spam Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #eef2ff;
            text-align: center;
            padding: 50px 15px;
        }
        .box {
            background: white;
            max-width: 500px;
            margin: auto;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px #cccccc;
        }
        textarea {
            width: 95%;
            height: 120px;
            padding: 10px;
            margin: 15px 0;
            box-sizing: border-box;
        }
        button {
            background: #4338ca;
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .result {
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="box">
        <h1>SMS Spam Detector</h1>
        <p>Enter a message to check it.</p>

        <form method="POST">
            <textarea name="message"
                placeholder="Type your message here..."
                required>{{ message }}</textarea>
            <br>
            <button type="submit">Check Message</button>
        </form>

        {% if result %}
            <div class="result">Result: {{ result }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    message = ""
    result = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            prediction = model.predict([message])[0]

            if prediction == "spam":
                result = "SPAM MESSAGE"
            else:
                result = "NORMAL MESSAGE"

    return render_template_string(
        HTML, message=message, result=result
    )

if __name__ == "__main__":
    app.run(debug=True)
