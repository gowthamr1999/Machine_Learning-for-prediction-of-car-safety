from flask import Flask,render_template,request,redirect
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,preprocessing
app = Flask(__name__)

app.config['SECRET_KEY'] = '86952dcd4291729cfe42eea4e46382f6'

@app.route("/")
@app.route("/sign-up", methods=["GET", "POST"])
def sign_up():
    if request.method == "POST":
        req = request.form
        data = request.form.to_dict(flat=False)
        buy = request.form["buy"]
        maintain = request.form["maintain"]
        door = request.form["door"]
        person = request.form["person"]
        boot = request.form["boot"]
        safe = request.form["safe"]
        inputs = [buy,maintain,door,person,boot,safe]
        values = pd.DataFrame([[buy,maintain,door,person,boot,safe]])
        result=predict(values,inputs)
        return render_template("user.html",result1=result,inputs=inputs,request=data)
    return render_template("sign-up.html")

def predict(values,inputs):
	data = pd.read_csv("car.csv")
	le = preprocessing.LabelEncoder()
	buying = le.fit_transform(list(data["buying"]))
	maint = le.fit_transform(list(data["maint"]))
	door = le.fit_transform(list(data["door"]))
	persons = le.fit_transform(list(data["persons"]))
	lug_boot = le.fit_transform(list(data["lug_boot"]))
	safety = le.fit_transform(list(data["safety"]))
	cls = le.fit_transform(list(data["class"]))
	predict = 'class'
	X = list(zip(buying, maint, door, persons, lug_boot, safety))
	y = list(cls)
	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
	model = KNeighborsClassifier(n_neighbors=9)
	model.fit(x_train, y_train)
	acc = model.score(x_test, y_test)
	predicted = model.predict(values)
	names = ["bad","okok","good","very-good"]
	for x in range(len(predicted)):
		return(names[predicted[x]])

if __name__=='__main__':
	app.run(debug=True)
    