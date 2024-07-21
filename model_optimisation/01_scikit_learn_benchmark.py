__author__ = "eolus87"

#%% Libraries
# Standard libraries
import time
from datetime import datetime

# Third party libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
# Custom libraries


# %% Setting up the experiment
number_of_experiments = 10
training_times = []
inference_times = []

#%% Data generation
data = pd.read_csv('winequality-white.csv', sep=';')
# %% Selecting features and targets and splitting the dataset
X = data.drop('quality', axis=1)  # Features: all columns except 'quality'
y = data['quality']                # Target variable: 'quality'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Model training
for i in range(number_of_experiments):
    svr_rbf = SVR(kernel='rbf', C=1000.0, epsilon=0.001, gamma='scale')
    tic = time.time()
    print(f"Starting the training of model {i}")
    svr_rbf.fit(X_train, y_train)
    toc = time.time()
    training_times.append(toc - tic)
    print(f"Model trained, time spent: {training_times[-1]} s")
    time.sleep(0.1)

print(f"Training time: {training_times} s")

#%% Model Error
y_pred = svr_rbf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Validation Mean Absolute Error (MAE): {mae}")

#%% Inference time
for i in range(number_of_experiments):
    tic = time.time()
    print(f"Starting the inference of model {i}")
    svr_rbf.predict(data.drop('quality', axis=1))
    toc = time.time()
    inference_times.append(toc - tic)
    print(f"Inference done, time spent: {inference_times[-1]} s")
    time.sleep(0.1)

#%% Generating report
report_df = pd.DataFrame({
    'Experiment Number': range(1, number_of_experiments + 1),
    'Training Time (s)': training_times,
    'Inference Time (s)': inference_times
})

now = datetime.now()
formatted_now = now.strftime("%y%m%d_%H%M%S")

# Step 3: Save the DataFrame to a CSV file
report_df.to_csv(f'{formatted_now}_experiment_times_report_raw_skl.csv', index=False)

print("Report saved as experiment_times_report.csv")