from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
import joblib  # To load your trained model

app = Flask(__name__)

# Load your trained model (assuming it's saved as a .pkl file)
#model = joblib.load('heart_attack_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # This Python 3 environment comes with many helpful analytics libraries installed
        # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
        # For example, here's several helpful packages to load

        import numpy as np  # linear algebra
        import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

        # Input data files are available in the read-only "../input/" directory
        # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

        import os
        for dirname, _, filenames in os.walk('/kaggle/input'):
            for filename in filenames:
                print(os.path.join(dirname, filename))

        # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
        # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

        import warnings
        warnings.filterwarnings("ignore")
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = pd.read_csv("C:/Users/teodo/Desktop/Heart Attack Prediction Project/Heart Attack/Heart Attack/heart.csv")

        # df
        # print(type(df))

        # df.head()

        new_col = ["age", "sex", "cp", "trtbps", "chol", "fbs", "rest_ecg", "thalach", "exang", "oldpeak", "slope",
                   "ca", "thal", "target"]

        df.columns = new_col

        # df.head()

        # print("Shape of Dataset: ", df.shape)

        # df.info()

        # df.isnull().sum()

        isnull_nr = []
        for i in df.columns:
            x = df[i].isnull().sum()
            isnull_nr.append(x)

        pd.DataFrame(isnull_nr, index=df.columns, columns=["Total Missing Values"])

        import missingno
        missingno.bar(df, color="b")

        # df.head()

        df["cp"].value_counts()

        df["cp"].value_counts().sum()

        df["cp"].value_counts().count()

        unique_nr = []
        for i in df.columns:
            x = df[i].value_counts().count()
            unique_nr.append(x)

        pd.DataFrame(unique_nr, index=df.columns, columns=["Total Unique Values"])

        # df.head()

        numeric_var = ["age", "trtbps", "chol", "thalach", "oldpeak"]
        categoric_var = ["sex", "cp", "fbs", "rest_ecg", "exang", "slope", "ca", "thal", "target"]

        df[numeric_var].describe()

        sns.distplot(df["age"], hist_kws=dict(linewidth=1, edgecolor="k"))

        sns.distplot(df["trtbps"], hist_kws=dict(linewidth=1, edgecolor="k"), bins=20)

        sns.distplot(df["chol"], hist=False)

        x, y = plt.subplots(figsize=(8, 6))
        sns.distplot(df["thalach"], hist=False, ax=y)
        y.axvline(df["thalach"].mean(), color="r", ls='--')

        x, y = plt.subplots(figsize=(8, 6))
        sns.distplot(df["oldpeak"], hist_kws=dict(linewidth=1, edgecolor="k"), bins=20, ax=y)
        y.axvline(df["oldpeak"].mean(), color="r", ls="--")

        numeric_var

        numeric_axis_name = ["Age of the Patient", "Resting Blood Pressure", "Cholesterol",
                             "Maximum Heart Rate Achieved", "ST Depression"]

        list(zip(numeric_var, numeric_axis_name))

        # title_font = {"family":"ariel", "color":"red", "weight":"bold","size":14}
        # axis_font  = {"family":"ariel", "color":"blue", "weight":"bold","size":12}
        # for i,z in list(zip(numeric_var, numeric_axis_name)):
        #   plt.figure(figsize=(8,6),dpi=80)
        #   sns.distplot(df[i], hist_kws = dict(linewidth=1, edgecolor="k"),bins=20)
        #   plt.title(i, fontdict=title_font)
        #   plt.xlabel(z, fontdict=axis_font)
        #   plt.ylabel("Density",fontdict=axis_font)

        #   plt.tight_layout()
        # plt.show()

        categoric_var

        categoric_axis_name = ["Gender", "Chest Pain Type", "Fasting Blood sugar",
                               "Resting Electrocardiographic Results",
                               "Exercise Induced Angina", "The Slope of ST Segment", "Number of Major Vessels", "Thal",
                               "Target"]

        list(zip(categoric_var, categoric_axis_name))

        df["cp"].value_counts()

        list(df["cp"].value_counts())

        list(df["cp"].value_counts().index)

        title_font = {"family": "arial", "color": "red", "weight": "bold", "size": 14}
        axis_font = {"family": "arial", "color": "blue", "weight": "bold", "size": 12}

        for i, z in list(zip(categoric_var, categoric_axis_name)):
            fig, ax = plt.subplots(figsize=(8, 6))

            observation_values = list(df[i].value_counts().index)
            total_observation_values = list(df[i].value_counts())
            ax.pie(total_observation_values, labels=observation_values, autopct='%1.1f%%', startangle=110,
                   labeldistance=1.1)
            ax.axis("equal")
            plt.title((i + "(" + z + ")"), fontdict=title_font)
            plt.legend()
            # plt.show()

        df[df["thal"] == 0]

        df["thal"] = df["thal"].replace(0, np.nan)

        df.loc[[48, 281], :]

        isnull_nr = []
        for i in df.columns:
            x = df[i].isnull().sum()
            isnull_nr.append(x)

        pd.DataFrame(isnull_nr, index=df.columns, columns=["Total Missing Values"])

        df["thal"].fillna(2, inplace=True)

        df.loc[[48, 281], :]

        df

        df["thal"] = pd.to_numeric(df["thal"], downcast="integer")

        df.loc[[48, 281], :]

        isnull_nr = []
        for i in df.columns:
            x = df[i].isnull().sum()
            isnull_nr.append(x)

        pd.DataFrame(isnull_nr, index=df.columns, columns=["Total Missing Values"])

        df["thal"].value_counts()

        from sklearn.preprocessing import RobustScaler

        df.head()

        df.drop(["chol", "fbs", "rest_ecg"], axis=1, inplace=True, errors='ignore')

        df.head()

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))
        ax1.boxplot(df["age"])
        ax1.set_title("age")
        ax2.boxplot(df["trtbps"])
        ax2.set_title("trtbps")
        ax3.boxplot(df["thalach"])
        ax3.set_title("thalach")
        ax4.boxplot(df["oldpeak"])
        ax4.set_title("oldpeak")

        # plt.show()

        from scipy import stats
        from scipy.stats import zscore
        from scipy.stats.mstats import winsorize

        z_trtbps = zscore(df["trtbps"])
        # for threshold in range (1,4):
        #   print("Threshold Value: {}".format(threshold))
        #  print("Number of Outliers: {}".format(len(np.where(z_trtbps > threshold)[0])))
        # print("----------------------")

        df[z_trtbps > 2][["trtbps"]]

        df[z_trtbps > 2][["trtbps"]].min()

        df[df["trtbps"] < 170].trtbps.max()

        winsorize_percentile_trtbps = (stats.percentileofscore(df["trtbps"], 165)) / 100
        # print(winsorize_percentile_trtbps)

        1 - winsorize_percentile_trtbps

        trtbps_winsorize = winsorize(df.trtbps, (0, (1 - winsorize_percentile_trtbps)))

        plt.boxplot(trtbps_winsorize)
        plt.xlabel("trtbps_winsorize", color="b")
        # plt.show()

        df["trtbps_winsorize"] = trtbps_winsorize

        df.head()

        def iqr(df, var):
            q1 = np.quantile(df[var], 0.25)
            q3 = np.quantile(df[var], 0.75)
            diff = q3 - q1
            lower_v = q1 - (1.5 * diff)
            upper_v = q3 + (1.5 * diff)
            return df[(df[var] < lower_v) | (df[var] > upper_v)]

        thalach_out = iqr(df, "thalach")

        thalach_out

        df.drop([272], axis=0, inplace=True)

        df["thalach"][270:275]

        plt.boxplot(df['thalach'])

        def iqr(df, var):
            q1 = np.quantile(df[var], 0.25)
            q3 = np.quantile(df[var], 0.75)
            diff = q3 - q1
            lower_v = q1 - (1.5 * diff)
            upper_v = q3 + (1.5 * diff)
            return df[(df[var] < lower_v) | (df[var] > upper_v)]

        iqr(df, "oldpeak")

        df[df["oldpeak"] < 4.2].oldpeak.max()

        win_per_oldpeak = (stats.percentileofscore(df["oldpeak"], 4)) / 100
        # print(win_per_oldpeak)

        oldpeak_win = winsorize(df.oldpeak, (0, (1 - win_per_oldpeak)))

        plt.boxplot(oldpeak_win)
        plt.xlabel("oldpeak_winsorize", color="b")
        # plt.show()

        df["oldpeak_winsorize"] = oldpeak_win

        df.head()

        df.drop(["trtbps", "oldpeak"], axis=1, inplace=True)

        df.head()

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))
        ax1.hist(df["age"])
        ax1.set_title("age")
        ax2.hist(df["trtbps_winsorize"])
        ax2.set_title("trtbps_winsorize")
        ax3.hist(df["thalach"])
        ax3.set_title("thalach")
        ax4.hist(df["oldpeak_winsorize"])
        ax4.set_title("oldpeak_winsorize")

        # plt.show()

        df[["age", "trtbps_winsorize", "thalach", "oldpeak_winsorize"]].agg(["skew"]).transpose()

        df["oldpeak_winsorize_log"] = np.log(df["oldpeak_winsorize"])
        df["oldpeak_winsorize_sqrt"] = np.sqrt(df["oldpeak_winsorize"])

        df.head()

        df[["oldpeak_winsorize", "oldpeak_winsorize_log", "oldpeak_winsorize_sqrt"]].agg(["skew"]).transpose()

        df.drop(["oldpeak_winsorize", "oldpeak_winsorize_log"], axis=1, inplace=True)

        #######################################################print(df.head().columns)

        df_copy = df.copy()

        df_copy.head()

        categoric_var

        categoric_var.remove("fbs")
        categoric_var.remove("rest_ecg")

        categoric_var

        # df_copy = pd.get_dummies(df_copy, columns = categoric_var[:-1], drop_first=True)


        new_numeric_var = ["age", "thalach", "trtbps_winsorize", "oldpeak_winsorize_sqrt"]

        robust_scaler = RobustScaler()

        df_copy[new_numeric_var] = robust_scaler.fit_transform(df_copy[new_numeric_var])

        df_copy.head()

        ##################################################################################################################
        ##################################################################################################################

        from sklearn.model_selection import train_test_split

        X_train = df_copy.drop(["target"], axis=1)
        y_train = df_copy[["target"]]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3)

        print(X_train.columns)

        y_train.head()

        # print(f"X_train: {X_train.shape[0]}")
        # print(f"X_test: {X_test.shape[0]}")
        # print(f"y_train: {y_train.shape[0]}")
        # print(f"y_test: {y_test.shape[0]}")

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        log_reg = LogisticRegression()
        log_reg

        log_reg.fit(X_train, y_train)

        # y_pred = log_reg.predict(X_test)

        # y_pred

        # accuracy = accuracy_score(y_test, y_pred)
        # print("Test accuracy:{}".format(accuracy))

        from sklearn.model_selection import cross_val_score

        # scores = cross_val_score(log_reg, X_test, y_test, cv = 10)
        # print("Cross-Validation Accuracy Scores", scores.mean())

        from sklearn.metrics import RocCurveDisplay

        # RocCurveDisplay.from_estimator(log_reg, X_test, y_test, name="Logistic Regression")
        # plt.title("Logistic Regression ROC Curve and AUC")
        # plt.plot([0, 1], [0, 1], "r--")
        # plt.show()

        from sklearn.model_selection import GridSearchCV

        log_reg_new = LogisticRegression()
        log_reg_new

        parameters = {"penalty": ["l1", "l2"], "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

        log_reg_grid = GridSearchCV(log_reg_new, param_grid=parameters)

        log_reg_grid.fit(X_train, y_train)

        # print("Best perameters:", log_reg_grid.best_params_)

        log_reg_2 = LogisticRegression(penalty="l1", solver="saga")
        log_reg_2

        log_reg_2.fit(X_train, y_train)

        # Get the form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        trtbps = int(request.form['trtbps'])
        oldpeak = float(request.form['oldpeak'])

        # Prepare the data (just like in your original code)
        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'thalach': thalach,
            'exang': exang,
            'slope': slope,
            'ca': ca,
            'thal': thal,
            'trtbps_winsorize': trtbps,
            'oldpeak_winsorize_sqrt': oldpeak
        }
        df_input = pd.DataFrame([input_data])
        # Predict using the model
        prediction = log_reg.predict(df_input)[0]

        # Return the result as a JSON response
        result = {'prediction': 'High risk' if prediction == 1 else 'Low risk'}
        return jsonify(result)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
