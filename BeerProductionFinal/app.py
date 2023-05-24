from flask import Flask, request, render_template
import pandas as pd
from numpy import array
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def increase_month(date_string):#increase the months by 1
    date_obj = datetime.strptime(date_string, "%Y-%m")
    new_date_obj = date_obj + relativedelta(months=1)
    formatted_date = datetime.strftime(new_date_obj, "%Y-%m")
    return formatted_date


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("beer_prod.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    prediction = int(request.form['prediction'])*12#as the input is in yesr and the model is on months
    df = pd.read_csv('last10Data.csv')#taking this data to take some previous data in the model
    data = df['value2'].tolist()#this column has the scaler data
    forecasted_df = pd.DataFrame(columns=['Month', 'value2'])#taking a dataframe to save the predictions
    start_month = '1995-08'#this month the predictions is ended
    scaler = MinMaxScaler().fit(df['value'].values.reshape(-1, 1))#fited the value in minmaxscaler to inverse transform

    import tensorflow as tf

    model = tf.keras.models.load_model('my_lstm_model')#load the model

    steps = 10
    features = 1

    for i in range(prediction):#as there have the multiple value so i took this data in a loop and predict for each month
        new_month = increase_month(start_month)
        forecast_input = array(data[-10:])
        x_forecast = forecast_input.reshape(1, steps, features)
        forecasted_y = model.predict(x_forecast).item()
        forecasted_df.at[i, 'Month'] = new_month
        forecasted_df.at[i, 'value2'] = forecasted_y
        data.append(forecasted_y)
        start_month = new_month

    forcasted_dif = scaler.inverse_transform(np.array(forecasted_df['value2']).reshape(-1, 1))#the scaller value is heere for reverse transform

    dif_list = np.array(forcasted_dif).flatten().tolist()

    last_value = df.tail(1)['production'].iloc[0]

    forecasted_og = []

    for i in range(len(dif_list)):#taking original value
        n = last_value + dif_list[i]
        forecasted_og.append(n)
        last_value = n

    forecasted_df['Production'] = forecasted_og

    forecasted_df = forecasted_df.drop('value2', axis=1)

    data = forecasted_df.to_json(orient='records')#make outputs as a json file

    return render_template('beer_prod.html', data=data)


if __name__ == '__main__':
    app.run(debug=True, port=6298)
