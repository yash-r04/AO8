from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/benchmark')
def benchmark():
    return render_template('benchmark.html')

@app.route('/report')
def report():
    return render_template('report.html')

if __name__ == '__main__':
    app.run(debug=True)
