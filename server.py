from flask import Flask, send_from_directory, render_template
import os
app = Flask(__name__, static_url_path=os.path.join(os.getcwd(), 'static'))
app._static_folder = os.path.abspath("static/")
print(app._static_folder)

# @app.route("/node_modules/recorderjs/recorder.js")
@app.route("/")
def root():
    # return app.send_static_file('index.html')
    # return "Hello"
    root_dir = os.getcwd()
    return send_from_directory(os.path.join(root_dir, "static"), "index.html")
    # return render_template('/Users/yuchen_zhang/workspace/build/tf/hackathon/static/index.html')