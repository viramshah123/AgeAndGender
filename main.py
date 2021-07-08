from flask import Flask, render_template,request

import gad1

app = Flask(__name__)


@app.route('/')
def index():
   return render_template('age_and_gender.html')


@app.route('/FlaskTutorial',  methods=['POST'])
def success():
   if request.method == 'POST':
       return gad1.gad_call()
   else:
       pass
if __name__ == "__main__":
   app.run()