import random

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
import joblib  # To load your trained model

app = Flask(__name__)


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

        new_col = ["age", "sex", "cp", "trtbps", "chol", "fbs", "rest_ecg", "thalach", "exang", "oldpeak", "slope",
                   "ca", "thal", "target"]

        df.columns = new_col

        isnull_nr = []
        for i in df.columns:
            x = df[i].isnull().sum()
            isnull_nr.append(x)

        pd.DataFrame(isnull_nr, index=df.columns, columns=["Total Missing Values"])

        import missingno
        missingno.bar(df, color="b")

        unique_nr = []
        for i in df.columns:
            x = df[i].value_counts().count()
            unique_nr.append(x)

        pd.DataFrame(unique_nr, index=df.columns, columns=["Total Unique Values"])

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

        numeric_axis_name = ["Age of the Patient", "Resting Blood Pressure", "Cholesterol",
                             "Maximum Heart Rate Achieved", "ST Depression"]

        list(zip(numeric_var, numeric_axis_name))

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

        df["thal"] = pd.to_numeric(df["thal"], downcast="integer")

        df.loc[[48, 281], :]

        isnull_nr = []
        for i in df.columns:
            x = df[i].isnull().sum()
            isnull_nr.append(x)

        pd.DataFrame(isnull_nr, index=df.columns, columns=["Total Missing Values"])

        df["thal"].value_counts()

        from sklearn.preprocessing import RobustScaler
        df.drop(["chol", "fbs", "rest_ecg"], axis=1, inplace=True, errors='ignore')

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
        import openai
        import os
        from groq import Groq

        z_trtbps = zscore(df["trtbps"])
        df[z_trtbps > 2][["trtbps"]]

        df[z_trtbps > 2][["trtbps"]].min()

        df[df["trtbps"] < 170].trtbps.max()

        winsorize_percentile_trtbps = (stats.percentileofscore(df["trtbps"], 165)) / 100

        1 - winsorize_percentile_trtbps

        trtbps_winsorize = winsorize(df.trtbps, (0, (1 - winsorize_percentile_trtbps)))

        plt.boxplot(trtbps_winsorize)
        plt.xlabel("trtbps_winsorize", color="b")

        df["trtbps_winsorize"] = trtbps_winsorize

        def iqr(df, var):
            q1 = np.quantile(df[var], 0.25)
            q3 = np.quantile(df[var], 0.75)
            diff = q3 - q1
            lower_v = q1 - (1.5 * diff)
            upper_v = q3 + (1.5 * diff)
            return df[(df[var] < lower_v) | (df[var] > upper_v)]

        thalach_out = iqr(df, "thalach")
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
        oldpeak_win = winsorize(df.oldpeak, (0, (1 - win_per_oldpeak)))

        plt.boxplot(oldpeak_win)
        plt.xlabel("oldpeak_winsorize", color="b")

        df["oldpeak_winsorize"] = oldpeak_win
        df.drop(["trtbps", "oldpeak"], axis=1, inplace=True)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))
        ax1.hist(df["age"])
        ax1.set_title("age")
        ax2.hist(df["trtbps_winsorize"])
        ax2.set_title("trtbps_winsorize")
        ax3.hist(df["thalach"])
        ax3.set_title("thalach")
        ax4.hist(df["oldpeak_winsorize"])
        ax4.set_title("oldpeak_winsorize")
        df[["age", "trtbps_winsorize", "thalach", "oldpeak_winsorize"]].agg(["skew"]).transpose()

        df["oldpeak_winsorize_log"] = np.log(df["oldpeak_winsorize"])
        df["oldpeak_winsorize_sqrt"] = np.sqrt(df["oldpeak_winsorize"])
        df[["oldpeak_winsorize", "oldpeak_winsorize_log", "oldpeak_winsorize_sqrt"]].agg(["skew"]).transpose()

        df.drop(["oldpeak_winsorize", "oldpeak_winsorize_log"], axis=1, inplace=True)

        df_copy = df.copy()
        categoric_var.remove("fbs")
        categoric_var.remove("rest_ecg")
        new_numeric_var = ["age", "thalach", "trtbps_winsorize", "oldpeak_winsorize_sqrt"]

        robust_scaler = RobustScaler()

        df_copy[new_numeric_var] = robust_scaler.fit_transform(df_copy[new_numeric_var])

        from sklearn.model_selection import train_test_split

        X_train = df_copy.drop(["target"], axis=1)
        y_train = df_copy[["target"]]
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import RocCurveDisplay
        from sklearn.model_selection import GridSearchCV

        log_reg_new = LogisticRegression()
        parameters = {"penalty": ["l1", "l2"], "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

        log_reg_grid = GridSearchCV(log_reg_new, param_grid=parameters)

        log_reg_grid.fit(X_train, y_train)
        log_reg_2 = LogisticRegression(penalty="l1", solver="saga")
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
        # 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
       # client = Groq(api_key=os.environ.get("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"))
       # client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        high_risk = [
            "As you are at high risk of a heart attack, staying active is crucial. Aim for at least 150 minutes of moderate exercise each week.",
            "Given your high risk, focus on a heart-healthy diet rich in fruits, vegetables, whole grains, and lean proteins.",
            "Limit your intake of salt, sugar, and saturated fats to help manage your risk of heart attack.",
            "Stay hydrated! Drinking plenty of water is important, especially with your high risk.",
            "Regularly monitor your blood pressure, as it’s essential to keep it within a healthy range given your situation.",
            "Avoid smoking and limit alcohol consumption to significantly reduce your heart attack risk.",
            "Manage stress through relaxation techniques like meditation or yoga, especially since stress can impact heart health.",
            "Make sure to get regular check-ups with your healthcare provider to stay informed about your health, considering your high risk.",
            "Maintaining a healthy weight is important for you; balance calories in with calories out.",
            "Stay connected with loved ones for support, as mental health plays a vital role in managing your heart health risk."
        ]

        low_risk = [
            "You have a low risk of heart attack! Keep up your active lifestyle to maintain this.",
            "With your low risk, continue enjoying a balanced diet rich in fruits, vegetables, and whole grains for heart health.",
            "Since you have low risk, stay mindful of your salt and sugar intake to keep it that way.",
            "Great job staying hydrated! Drinking plenty of water supports your overall health and heart.",
            "Regular check-ups are still important to monitor your health, even with your low risk.",
            "Celebrate your healthy habits! Not smoking and moderating alcohol consumption help keep your risk low.",
            "Stay active! Regular exercise is key to maintaining your low risk of heart attack.",
            "Even with low risk, manage stress effectively, as it can impact your heart health over time.",
            "Maintaining a healthy weight is important to support your well-being and keep your risk low.",
            "Keep nurturing your social connections; they’re beneficial for your mental and heart health!"
        ]

        if prediction == 1:
            prompt=random.choice(high_risk)
            #prompt="Generate a short message for someone at high risk of heart attack. Please keep it general and do not include any names. Limit the message to 2-3 sentences."
        else:
            prompt=random.choice(low_risk)
            #prompt="Generate a short message for someone at low risk of heart attack. Please keep it general and do not include any names. Limit the message to 2-3 sentences."
        '''
        chat_completion = client.chat.completions.create(
        messages=[
        {
        "role": "user",
        "content": prompt,
        } ],
        model = "llama3-8b-8192",
        )
        res= chat_completion.choices[0].message.content.strip()
        '''
        result = {'prediction': prompt}

        return jsonify(result)


    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


