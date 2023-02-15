import flask
import pickle
import pandas as pd
import numpy as np

# Use pickle to load in the pre-trained model.
with open(f'model/model_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

with open(f'model/model_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict(input_data):
    input_data_scaled = np.zeros(19)
    input_data_scaled[0] = input_data.iloc[0][0]
    input_data_scaled[1] = input_data.iloc[0][1]
    input_data_scaled[5] = input_data.iloc[0][2]
    input_data_scaled[6] = input_data.iloc[0][3]
    input_data_scaled[8] = input_data.iloc[0][4]
    input_data_scaled[11] = input_data.iloc[0][5]
    input_data_scaled = input_data_scaled.reshape(1,19)
    input_data_scaled = scaler.transform(input_data_scaled)
    l=[]
    for j in [0,1,5,6,8,11]:
        l.append(input_data_scaled[0][j])
    prediction = model.predict([l])[0]
    if prediction == 1 :
        prediction = 'YES'
    else: 
        prediction = 'NO'
    return prediction

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        gp = flask.request.form['GP']
        minn = flask.request.form['MIN']
        fg = flask.request.form['FG%']
        pm = flask.request.form['3P Made']
        p = flask.request.form['3P%']
        ft = flask.request.form['FT%']
        input_variables = pd.DataFrame([[gp, minn, fg,pm,p,ft]],
                                       columns=['gp', 'minn', 'fg%','3pm','3p%','ft%'],
                                       dtype=float)
        prediction = predict(input_variables)
        return flask.render_template('main.html',
                                     original_input={'GP':gp,
                                                     'MIN':minn,
                                                     'FG%':fg,
                                                     '3P Made':pm,
                                                     '3P%':p,
                                                     'FT%':ft},
                                     result=prediction,
                                     )
    

if __name__ == '__main__':
    app.run()