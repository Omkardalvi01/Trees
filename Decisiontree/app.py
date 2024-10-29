from flask import Flask, render_template, request
import pickle
import pandas as pd

model, lbl_encoder, one_hot_encoder = pickle.load(open('clf.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    data5 = float(request.form['e'])
    data6 = request.form['f']
    data7 = request.form['g']

    # Prepare the continuous and categorical dataframes
    cont_df = pd.DataFrame([[data1, data2, data3, data4, data5]], columns=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year'])
    cat_df = pd.DataFrame([[data6, data7]], columns=['island', 'sex'])
    
    one_hot = one_hot_encoder.transform(cat_df)
    one_hot_df = pd.DataFrame(one_hot, columns=one_hot_encoder.get_feature_names_out(['island', 'sex']))

    df = pd.concat([cont_df, one_hot_df], axis=1)
    
    pred = model.predict(df)
    result = lbl_encoder.inverse_transform(pred)
    return render_template('after.html', data=result[0])

if __name__ == "__main__":
    app.run(debug=True)
