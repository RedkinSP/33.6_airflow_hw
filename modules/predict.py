import dill
import os
import pandas as pd
from datetime import datetime


path = os.environ.get('PROJECT_PATH', '.')

model_path = f'{path}/data/models/'
model_filename = os.listdir(model_path)[-1]
model_fullname = model_path + model_filename

with open(model_fullname, 'rb') as file:
    model = dill.load(file)

def predict():
    test_path = f'{path}/data/test/'
    test_files = os.listdir(test_path)

    results = []
    for json_file in test_files:
        filename = test_path + json_file
        json = pd.read_json(filename, orient='index').T
        pred = model.predict(json)
        results.append([json['id'][0], pred[0]])
    prediction = pd.DataFrame(data=results, columns=['car_id', 'pred'])
    print(prediction)

    csv_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    prediction.to_csv(csv_filename, index=False)

if __name__ == '__main__':
    predict()
