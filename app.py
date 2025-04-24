from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
# app = Flask(__name__, template_folder="path_to_your_templates_folder")


model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    data1 = int(float(request.form['a']))
    data2 = int(float(request.form['b']))
    data3 = int(float(request.form['c']))
    arr = np.array([[data1, data2, data3]])
    output = model.predict(arr)

    # def to_str(var):
    #     return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

    # def to_str(var):
    #     return str(int(np.reshape(np.asarray(var), (1, np.size(var)))[0]))

    def to_str(var):
        return str(int(np.asarray(var).flatten()[0]))



    if (output < 4):
        risk = 'No'
    elif (4 <= output < 6):
        risk = 'Low'
    elif (6 <= output < 8):
        risk = 'Moderate'
    elif (8 <= output < 9):
        risk = 'High'
    else:
        risk = 'Very High'

    return render_template('prediction.html', p=to_str(output), q=risk)

if __name__ == "__main__":
    app.run(debug=True)
