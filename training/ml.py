from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import pickle

app = Flask(__name__)

# Load your dataset
df = pd.read_csv('COMP.csv', encoding='ISO-8859-1')
df = df.dropna()

# Feature selection (excluding 'plugin', 'codec', 'level')
features = df.columns.difference(['plugin', 'codec', 'level'])

# Split the data into training and testing sets
X_train, X_test, y_plugin_train, y_plugin_test, y_codec_train, y_codec_test, y_level_train, y_level_test = train_test_split(
    df[features], df['plugin'], df['codec'], df['level'], test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Choose models (Decision Tree Classifiers)
model_plugin = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', DecisionTreeClassifier())])
model_codec = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', DecisionTreeClassifier())])
model_level = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', DecisionTreeClassifier())])

# Train the models
model_plugin.fit(X_train, y_plugin_train)
model_codec.fit(X_train, y_codec_train)
model_level.fit(X_train, y_level_train)

# Save the models using pickle
with open('model_plugin.pkl', 'wb') as model_file:
    pickle.dump(model_plugin, model_file)

with open('model_codec.pkl', 'wb') as model_file:
    pickle.dump(model_codec, model_file)

with open('model_level.pkl', 'wb') as model_file:
    pickle.dump(model_level, model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendation.html', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        user_input = {}
        for feature in ['file-type', 'file-size', 'processor']:
            user_input[feature] = request.form[feature]

        user_df = pd.DataFrame([user_input], columns=features)

        # Load models from pickle files
        with open('model_plugin.pkl', 'rb') as model_file:
            model_plugin = pickle.load(model_file)

        with open('model_codec.pkl', 'rb') as model_file:
            model_codec = pickle.load(model_file)

        with open('model_level.pkl', 'rb') as model_file:
            model_level = pickle.load(model_file)

        # Make predictions for the user input
        predicted_plugin = model_plugin.predict(user_df)
        predicted_codec = model_codec.predict(user_df)
        predicted_level = model_level.predict(user_df)

        return render_template('results.html', 
                               predicted_plugin=predicted_plugin[0],
                               predicted_codec=predicted_codec[0],
                               predicted_level=predicted_level[0])

    return render_template('recommendation.html')

@app.route('/results.html', methods=['GET', 'POST'])
def results():
    # This route is for displaying results and should match the action attribute in the recommendation.html form
    return render_template('results.html')  # Add this line to return a response

if __name__ == '__main__':
    app.run(debug=True)
