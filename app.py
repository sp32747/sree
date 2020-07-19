import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import math

from sklearn.metrics import accuracy_score,classification_report, recall_score, precision_score,confusion_matrix

app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   # filename = 'finalized_model.pkl'
    test_df1 = pd.read_csv('test_FD001.txt', sep=" ", header=None)
    test_df1.drop(columns=[26,27], inplace= True)
    columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
    test_df1.columns = columns
    MachineID_name = ["unit_number"]
    RUL_name = ["time_in_cycles"]
    OS_name = columns[2:5]
    Sensor_name = columns[5:26]
    MachineID_data = test_df1[MachineID_name]
    RUL_data = test_df1[RUL_name]
    OS_data = test_df1[OS_name]
    Sensor_data = test_df1[Sensor_name]
    test_data = pd.concat([MachineID_data,RUL_data,OS_data,Sensor_data], axis=1)
    test_data.drop(test_data[["TRA","T2","P2","P15",
                "epr","farB","Nf_dmd","PCNfR_dmd"]], axis=1 , inplace=True)
    
    test = test_data.groupby('unit_number').max()
    result_df = pd.read_csv('RUL_FD001.txt', sep=" ", header=None)
    result_df.drop(columns = [1], axis=1, inplace= True)
    col = ["label"]
    result_df.columns = col
    result_df['label'] = result_df['label'].apply(lambda x: 1 if x <= 30 else 0)
    result_df.label.value_counts()
    res_true = result_df["label"]
  #  loaded_model = pickle.load(open(filename, 'rb'))
    res_pred_4 = model.predict(test)
    print(accuracy_score(res_true,res_pred_4))
    print(classification_report(res_true,res_pred_4))
    res = pd.DataFrame(res_pred_4)
    test1 = pd.DataFrame(test)
    test2 = test1.reset_index()
    j=0
    output_list=[]
    for i, item in enumerate(res[0]):
        if(item == 1): 
            j= j+1
            
            output_list.append("After 30 cycles Machine {} will breakdown".format(test2.unit_number[i]))
            print (" After 30 cycles Machine {} will breakdown".format(test2.unit_number[i]))
  
    print("Number of machins about to fail:",j)
    
    
    
    
    
    '''
    For rendering results on HTML GUI
    '''
    

    

    return render_template('index.html', prediction_list=output_list)


if __name__ == "__main__":
    app.run(debug=True)