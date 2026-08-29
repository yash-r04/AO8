# run.py
from app import create_app
from flask import render_template, session, redirect

app = create_app()

@app.route("/")
def dashboard():
    if "user" not in session:
        return redirect("/auth/login")
    return render_template("home.html", user=session["user"])

if __name__ == "__main__":
    app.run(debug=True, port=5000)