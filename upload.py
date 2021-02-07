import os,io
from flask import Flask, flash, request, redirect, url_for, Response, render_template, make_response
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import pandas as pd
import numpy as np
import sklearn
import array
import tempfile

UPLOAD_FOLDER = '/intern/upload'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#忘れたらflaskの公式見ればある
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            tempfile_path = tempfile.NamedTemporaryFile().name
            file.save(tempfile_path)
            df1=pd.read_csv(tempfile_path)
            return redirect(url_for('hello',
                                filename=df1))
         
        x2 = df1.drop(columns=['お仕事No.', '勤務地　最寄駅2（駅名）', '（紹介予定）雇用形態備考', '掲載期間　開始日','動画コメント','勤務地　最寄駅2（沿線名）','（派遣先）概要　勤務先名（漢字）','休日休暇　備考','（派遣）応募後の流れ','期間・時間　勤務時間','勤務地　備考','（紹介予定）入社時期','お仕事名','期間・時間　勤務開始日','（派遣先）勤務先写真ファイル名','（派遣先）配属先部署','動画タイトル','仕事内容','（派遣先）概要　事業内容','（紹介予定）年収・給与例','勤務地　最寄駅1（沿線名）','応募資格','（紹介予定）休日休暇','派遣会社のうれしい特典','掲載期間　終了日','お仕事のポイント（仕事PR）','動画ファイル名','（派遣先）職場の雰囲気','（紹介予定）待遇・福利厚生','勤務地　最寄駅1（駅名）','給与/交通費　備考','期間･時間　備考','拠点番号'])
          
          
        x4 = x2.dropna(axis=1)

        x6 = np.array(x4)
        x5 = np.array([[    3,     1,     0,     0,     0,     0,     0, 22010,     1,
                1,     1,     0,     1,     0,     1,     0,     1,     3,
                0,     1,     0,     0,     2,     1,     0,     1,     1,
                0,     0,     1,     1,     5,     0,     0,     0,     1,
                0,     0,     0,     1,     1,     0,     0,     1,     1,
            99,     0,     1,     0,     1,     0,     1,     0,     0,
                1,     1,     0,     1,     0,     0, 30000,    13,     0,
                0,     0,     0,     2,     0,     0,     1,     0,     1,
                1,     3,     1,     0,     1,     0,  1550,     1,     1,
                1,   121,     1],
        [    3,     1,     0,     1,     0,     0,     0, 20020,     1,
                1,     1,     0,     1,     0,     0,     0,     1,     3,
                0,     1,     0,     0,     2,     1,     0,     1,     1,
                0,     0,     1,     3,     5,     1,     0,     0,     1,
                0,     1,     0,     1,     1,     0,     0,     1,     1,
            99,     0,     1,     0,     1,     0,     1,     0,     0,
                1,     1,     0,     1,     0,     0, 28000,    13,     0,
                0,     0,     0,     1,     1,     0,     1,     0,     0,
                1,     3,     1,     0,     1,     0,  1800,     1,     1,
                1,   101,     1],
        [    3,     1,     0,     1,     0,     0,     0, 20020,     1,
                1,     1,     0,     1,     0,     0,     0,     1,     3,
                0,     1,     0,     0,     2,     1,     0,     1,     1,
                0,     0,     1,     3,     5,     1,     0,     0,     1,
                0,     1,     0,     1,     1,     0,     0,     1,     1,
            99,     0,     1,     0,     1,     0,     1,     0,     0,
                1,     1,     0,     1,     0,     0, 28000,    13,     0,
                0,     0,     0,     1,     1,     0,     1,     0,     0,
                1,     3,     1,     0,     1,     0,  1800,     1,     1,
                1,   101,     1],
        [    2,     1,     0,     0,     0,     0,     1, 20020,     0,
                1,     1,     0,     0,     0,     1,     0,     1,     3,
                0,     0,     0,     0,     2,     1,     0,     1,     1,
                0,     0,     1,     3,     5,     1,     0,     0,     1,
                0,     1,     0,     1,     1,     0,     0,     0,     1,
            99,     0,     1,     0,     1,     0,     1,     0,     0,
                1,     1,     0,     1,     0,     0, 30000,    13,     0,
                0,     0,     0,     1,     1,     0,     1,     0,     1,
                1,     3,     1,     0,     1,     0,  1650,     1,     1,
                1,   102,     0],
        [    2,     1,     0,     0,     0,     0,     1, 20020,     0,
                1,     1,     0,     0,     0,     1,     0,     1,     3,
                0,     0,     0,     0,     2,     1,     0,     1,     1,
                0,     0,     1,     3,     5,     1,     0,     0,     1,
                0,     1,     0,     1,     1,     0,     0,     0,     1,
            99,     0,     1,     0,     1,     0,     1,     0,     0,
                1,     1,     0,     1,     0,     0, 30000,    13,     0,
                0,     0,     0,     1,     1,     0,     1,     0,     1,
                1,     3,     1,     0,     1,     0,  1650,     1,     1,
                1,   102,     0],
        [    3,     1,     0,     1,     0,     0,     0, 20610,     0,
                1,     1,     0,     0,     0,     0,     0,     1,     3,
                0,     1,     0,     0,     2,     1,     0,     1,     1,
                0,     0,     1,     3,     5,     1,     0,     0,     1,
                0,     1,     0,     1,     1,     0,     0,     0,     1,
            99,     0,     1,     0,     1,     0,     1,     0,     0,
                1,     1,     0,     1,     0,     0, 22000,    13,     0,
                0,     0,     0,     4,     1,     0,     1,     0,     1,
                1,     3,     1,     0,     1,     0,  1600,     1,     1,
                1,   101,     1],
        [    3,     1,     0,     0,     0,     0,     0, 22010,     1,
                1,     1,     0,     0,     0,     1,     0,     1,     3,
                0,     1,     0,     0,     2,     1,     0,     1,     1,
                0,     0,     1,     3,     5,     0,     0,     0,     1,
                0,     0,     0,     1,     1,     0,     0,     0,     1,
            99,     0,     1,     0,     1,     0,     1,     1,     0,
                1,     1,     0,     1,     0,     0, 30000,    13,     0,
                0,     0,     0,     2,     0,     0,     1,     0,     1,
                1,     3,     1,     0,     1,     0,  1600,     1,     1,
                1,   112,     1],
        [    3,     1,     0,     0,     0,     0,     0, 22010,     1,
                1,     1,     0,     1,     0,     1,     0,     1,     3,
                0,     1,     0,     0,     2,     1,     0,     1,     1,
                0,     0,     1,     1,     5,     0,     0,     0,     1,
                0,     0,     0,     1,     1,     0,     0,     1,     1,
            99,     0,     1,     0,     1,     0,     1,     0,     0,
                1,     1,     0,     1,     0,     0, 30000,    13,     0,
                0,     0,     0,     3,     0,     0,     1,     0,     1,
                1,     3,     1,     0,     1,     0,  1700,     1,     1,
                1,   121,     1],
        [    3,     1,     0,     0,     0,     0,     0, 20020,     0,
                1,     1,     0,     1,     0,     1,     0,     1,     3,
                0,     1,     0,     0,     2,     1,     0,     1,     1,
                0,     0,     1,     3,     5,     1,     0,     0,     1,
                0,     0,     0,     1,     1,     0,     0,     0,     1,
            99,     0,     1,     0,     1,     0,     1,     1,     0,
                1,     1,     0,     1,     0,     0, 30000,    13,     0,
                0,     0,     0,     3,     1,     0,     1,     0,     1,
                1,     3,     1,     0,     1,     0,  1600,     1,     1,
                1,   107,     1],
        [    3,     1,     0,     0,     0,     0,     0, 20020,     0,
                1,     1,     0,     1,     0,     1,     0,     1,     3,
                0,     1,     0,     0,     2,     1,     0,     1,     1,
                0,     0,     1,     3,     5,     1,     0,     0,     1,
                0,     0,     0,     1,     1,     0,     0,     0,     1,
            99,     0,     1,     0,     1,     0,     1,     1,     0,
                1,     1,     0,     1,     0,     0, 30000,    13,     0,
                0,     0,     0,     3,     1,     0,     1,     0,     1,
                1,     3,     1,     0,     1,     0,  1600,     1,     1,
                1,   107,     1],
        [    3,     1,     0,     1,     0,     0,     0, 22030,     1,
                1,     1,     0,     1,     0,     0,     0,     1,     3,
                0,     1,     0,     0,     2,     1,     1,     1,     1,
                0,     0,     1,     3,     5,     1,     0,     0,     1,
                0,     1,     0,     1,     1,     0,     0,     1,     1,
            99,     0,     1,     0,     1,     0,     1,     0,     0,
                1,     1,     0,     1,     0,     0, 28000,    13,     0,
                0,     0,     0,     1,     1,     0,     1,     0,     0,
                1,     3,     1,     0,     1,     0,  1800,     1,     1,
                1,   101,     1],
        [    3,     1,     0,     1,     0,     0,     0, 22030,     1,
                1,     1,     0,     1,     0,     0,     0,     1,     3,
                0,     1,     0,     0,     2,     1,     1,     1,     1,
                0,     0,     1,     3,     5,     1,     0,     0,     1,
                0,     1,     0,     1,     1,     0,     0,     1,     1,
            99,     0,     1,     0,     1,     0,     1,     0,     0,
                1,     1,     0,     1,     0,     0, 28000,    13,     0,
                0,     0,     0,     1,     1,     0,     1,     0,     0,
                1,     3,     1,     0,     1,     0,  1800,     1,     1,
                1,   101,     1],
        [    3,     1,     0,     0,     0,     0,     1, 22020,     0,
                1,     1,     0,     1,     0,     0,     0,     1,     3,
                0,     0,     0,     0,     2,     1,     0,     1,     1,
                0,     0,     1,     3,     5,     1,     0,     0,     1,
                0,     1,     0,     1,     1,     0,     0,     0,     1,
            99,     0,     1,     0,     1,     0,     1,     0,     0,
                1,     1,     0,     1,     0,     0, 29000,    13,     0,
                0,     0,     0,     5,     1,     0,     1,     0,     0,
                1,     3,     1,     0,     1,     0,  1800,     1,     1,
                1,   101,     0],
        [    3,     1,     0,     0,     0,     0,     1, 22020,     0,
                1,     1,     0,     1,     0,     0,     0,     1,     3,
                0,     0,     0,     0,     2,     1,     0,     1,     1,
                0,     0,     1,     3,     5,     1,     0,     0,     1,
                0,     1,     0,     1,     1,     0,     0,     0,     1,
            99,     0,     1,     0,     1,     0,     1,     0,     0,
                1,     1,     0,     1,     0,     0, 29000,    13,     0,
                0,     0,     0,     5,     1,     0,     1,     0,     0,
                1,     3,     1,     0,     1,     0,  1800,     1,     1,
                1,   101,     0],
        [    3,     1,     0,     0,     0,     0,     0, 20030,     0,
                1,     1,     0,     1,     0,     0,     0,     1,     3,
                0,     1,     0,     0,     2,     1,     1,     1,     1,
                0,     0,     1,     3,     5,     1,     0,     0,     1,
                0,     1,     1,     0,     1,     0,     0,     0,     1,
            99,     0,     1,     0,     1,     0,     1,     1,     0,
                1,     1,     0,     1,     1,     0, 23000,    13,     0,
                0,     0,     0,     3,     1,     0,     1,     0,     1,
                1,     3,     1,     0,     1,     0,  1700,     1,     1,
                1,   101,     0]])

        train_y = np.array([1.94591015, 3.95124372, 3.29583687, 3.97029191, 3.78418963,
            1.94591015, 2.56494936, 1.94591015, 3.21887582, 2.7080502 ,
            3.21887582, 3.29583687, 3.40119738, 3.29583687, 3.09104245])


    
                
        from sklearn.ensemble import RandomForestRegressor
        rfr = RandomForestRegressor(random_state=0)
        rfr.fit(x5, train_y)

        y_pred = rfr.predict(x6)

        yyy = pd.DataFrame(y_pred)

        y1 = df1['お仕事No.']
        y2 = yyy.iloc[:,0]
        y3 = pd.DataFrame()
        y3['お仕事No.'] = y1
        y3['応募数 合計'] = y2

        response = make_response()
        response.data  = open('test.zip', "rb").read()
        response.headers['Content-Type'] = 'application/octet-stream'
        response.headers['Content-Disposition'] = 'attachment; filename=test.zip'
        

                
                

        

        
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/hello')
def hello():
    return '''
        <html><body>
        Hello. <a href="/getPlotCSV">Click me.</a>
        </body></html>
        '''

@app.route("/getPlotCSV")
def getPlotCSV():
    # with open("outputs/Adjacency.csv") as fp:
    #     csv = fp.read()
    csv = 'unfinished'
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=myanswer.csv"})

if __name__ == '__main__':
    app.debug = False
    app.run()
