# run.py
from app import create_app
from flask import render_template, session

app = create_app()


@app.route("/")
def home():
    """Public landing page. Always renders, regardless of login state."""
    return render_template("index.html", user=session.get("user"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)