#!/usr/bin/python

from flask import Flask, Response
app = Flask(__name__)

@app.route("/")
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
    csv = 'C:/intern/data/titanic_text.csv'
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=myanswer.csv"})


app.run(debug=True)