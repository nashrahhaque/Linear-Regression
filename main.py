import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load and preprocess the first dataset
df_train_1 = pd.read_csv('train-100-10.csv')
df_test_1 = pd.read_csv('test-100-10.csv')
df_train_1 = df_train_1.iloc[:, :-2]

# Load the second dataset (100 rows and 100 columns)
df_train_2 = pd.read_csv('train-100-100.csv')
df_test_2 = pd.read_csv('test-100-100.csv')

# Load the third dataset (1000 rows and 100 columns)
df_train_3 = pd.read_csv('train-1000-100.csv')
df_test_3 = pd.read_csv('test-1000-100.csv')

# Split the third dataset into smaller datasets (1000 rows and 100 columns)
df_train_50 = df_train_3[0:50]
df_train_100 = df_train_3[0:100]
df_train_150 = df_train_3[0:150]


def dataframe_to_matrix(df):

    # Add a column of ones to represent the intercept term
    df.insert(0, 'intercept', 1)

    # Extract features (X) and targets (Y)
    X_matrix_train = df.iloc[:, :-1].values
    Y_matrix_train = df.iloc[:, -1:].values

    return X_matrix_train, Y_matrix_train


# Define functions to convert datasets to matrices
def convert_dataset_to_matrices(train_df, test_df, split_df):
    x_train, y_train = dataframe_to_matrix(train_df)
    x_test, y_test = dataframe_to_matrix(test_df)
    x_split, y_split = dataframe_to_matrix(split_df)
    return x_train, y_train, x_test, y_test, x_split, y_split

# Training Data
x_matrix_train_1, y_matrix_train_1, x_matrix_test_1, y_matrix_test_1, x_matrix_train_50, y_matrix_train_50 = convert_dataset_to_matrices(df_train_1, df_test_1, df_train_50)
x_matrix_train_2, y_matrix_train_2, x_matrix_test_2, y_matrix_test_2, x_matrix_train_100, y_matrix_train_100 = convert_dataset_to_matrices(df_train_2, df_test_2, df_train_100)
x_matrix_train_3, y_matrix_train_3, x_matrix_test_3, y_matrix_test_3, x_matrix_train_150, y_matrix_train_150 = convert_dataset_to_matrices(df_train_3, df_test_3, df_train_150)


def weight_calculation(X_train, Y_train, lambda_end, lambda_start=0):
    """Calculates the weights of linear regression with lambda ranging from a to b"""

    lambda_values = np.arange(lambda_start, lambda_end + 1)
    weights_varying_lambda = []

    for lambda_value in lambda_values:
        XTX_plus_I_inv = np.linalg.inv(np.dot(X_train.T, X_train) + lambda_value * np.identity(X_train.shape[1]))
        weights = np.dot(XTX_plus_I_inv, X_train.T @ Y_train)
        weights_varying_lambda.append(weights.flatten())

    return np.transpose(np.array(weights_varying_lambda))


# Define lambda start and end
lambda_start = 0
lambda_end = 150

# Calculate weights for Dataset 1
weights_train_dataset_1 = weight_calculation(x_matrix_train_1, y_matrix_train_1, lambda_end=150, lambda_start=0)

# Calculate weights for Dataset 2
weights_train_dataset_2 = weight_calculation(x_matrix_train_2, y_matrix_train_2, lambda_end=150, lambda_start=0)

# Calculate weights for Dataset 2 (Lambda 1-150)
weights_train_dataset_2_lambda_1_150 = weight_calculation(x_matrix_train_2, y_matrix_train_2, lambda_end=150, lambda_start=1)

# Calculate weights for Dataset 3
weights_train_dataset_3 = weight_calculation(x_matrix_train_3, y_matrix_train_3, lambda_end=150, lambda_start=0)

# Calculate weights for Dataset 3 (Lambda 1-150)
weights_train_dataset_3_lambda_1_150 = weight_calculation(x_matrix_train_3, y_matrix_train_3, lambda_end=150, lambda_start=1)

# Calculate weights for Dataset 4 - Train-50(1000)-100
weights_train_dataset_50_100 = weight_calculation(x_matrix_train_50, y_matrix_train_50, lambda_end=150, lambda_start=0)

# Calculate weights for Dataset 4 - Train-50(1000)-100 (Lambda 1-150)
weights_train_dataset_50_100_lambda_1_150 = weight_calculation(x_matrix_train_50, y_matrix_train_50, lambda_end=150, lambda_start=1)

# Calculate weights for Dataset 5 - Train-100(1000)-100
weights_train_dataset_100_100 = weight_calculation(x_matrix_train_100, y_matrix_train_100, lambda_end=150, lambda_start=0)

# Calculate weights for Dataset 5 - Train-100(1000)-100 (Lambda 1-150)
weights_train_dataset_100_100_lambda_1_150 = weight_calculation(x_matrix_train_100, y_matrix_train_100, lambda_end=150, lambda_start=1)

# Calculate weights for Dataset 6 - Train-150(1000)-100
weights_train_dataset_150_100 = weight_calculation(x_matrix_train_150, y_matrix_train_150, lambda_end=150, lambda_start=0)

# Calculate weights for Dataset 6 - Train-150(1000)-100 (Lambda 1-150)
weights_train_dataset_150_100_lambda_1_150 = weight_calculation(x_matrix_train_150, y_matrix_train_150, lambda_end=150, lambda_start=1)

def mean_squared_error(X_train, weights, Y_train):
    Y_pred = np.dot(X_train, weights)
    sum_error = 0.0

    for i in range(len(Y_train)):
        Y_pred_error = Y_pred[i] - Y_train[i]
        sum_error += (Y_pred_error ** 2)

    MSE = sum_error / float(len(Y_train))
    return MSE

# Dataset 1
MSE_train_dataset_1 = mean_squared_error(x_matrix_train_1, weights_train_dataset_1, y_matrix_train_1)
MSE_test_dataset_1 = mean_squared_error(x_matrix_test_1, weights_train_dataset_1, y_matrix_test_1)

# Dataset 2
MSE_train_dataset_2 = mean_squared_error(x_matrix_train_2, weights_train_dataset_2, y_matrix_train_2)
MSE_test_dataset_2 = mean_squared_error(x_matrix_test_2, weights_train_dataset_2, y_matrix_test_2)

MSE_train_dataset_2_lambda_1_150 = mean_squared_error(x_matrix_train_2, weights_train_dataset_2_lambda_1_150, y_matrix_train_2)
MSE_test_dataset_2_lambda_1_150 = mean_squared_error(x_matrix_test_2, weights_train_dataset_2_lambda_1_150, y_matrix_test_2)

# Dataset 3
MSE_train_dataset_3 = mean_squared_error(x_matrix_train_3, weights_train_dataset_3, y_matrix_train_3)
MSE_test_dataset_3 = mean_squared_error(x_matrix_test_3, weights_train_dataset_3, y_matrix_test_3)

MSE_test_dataset_3_lambda_1_150 = mean_squared_error(x_matrix_test_3, weights_train_dataset_3_lambda_1_150, y_matrix_test_3)

# Dataset 4
MSE_train_dataset_50_100 = mean_squared_error(x_matrix_train_50, weights_train_dataset_50_100, y_matrix_train_50)
MSE_test_dataset_50_100 = mean_squared_error(x_matrix_test_3, weights_train_dataset_50_100, y_matrix_test_3)

MSE_train_dataset_50_100_lambda_1_150 = mean_squared_error(x_matrix_train_50, weights_train_dataset_50_100_lambda_1_150, y_matrix_train_50)
MSE_test_dataset_50_100_lambda_1_150 = mean_squared_error(x_matrix_test_3, weights_train_dataset_50_100_lambda_1_150, y_matrix_test_3)

# Dataset 5
MSE_train_dataset_100_100 = mean_squared_error(x_matrix_train_100, weights_train_dataset_100_100, y_matrix_train_100)
MSE_test_dataset_100_100 = mean_squared_error(x_matrix_test_3, weights_train_dataset_100_100, y_matrix_test_3)




MSE_train_dataset_100_100_lambda_1_150 = mean_squared_error(x_matrix_train_100, weights_train_dataset_100_100_lambda_1_150, y_matrix_train_100)
MSE_test_dataset_100_100_lambda_1_150 = mean_squared_error(x_matrix_test_3, weights_train_dataset_100_100_lambda_1_150, y_matrix_test_3)

# Dataset 6
MSE_train_dataset_150_100 = mean_squared_error(x_matrix_train_150, weights_train_dataset_150_100, y_matrix_train_150)
MSE_test_dataset_150_100 = mean_squared_error(x_matrix_test_3, weights_train_dataset_150_100, y_matrix_test_3)

# Define a color palette
colors = ['#1f77b4', '#ff7f0e']

# Dataset 1
plt.plot(MSE_train_dataset_1, label='Train MSE', color=colors[0])
plt.plot(MSE_test_dataset_1, label='Test MSE', color=colors[1])
plt.title("Dataset 1:\n Train-100-10 vs Test-100-10\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Dataset 2
plt.plot(MSE_train_dataset_2, label='Train MSE', color=colors[0])
plt.plot(MSE_test_dataset_2, label='Test MSE', color=colors[1])
plt.title("Dataset 2:\n Train-100-100 vs Test-100-100\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Dataset 3
plt.plot(MSE_train_dataset_3, label='Train MSE', color=colors[0])
plt.plot(MSE_test_dataset_3, label='Test MSE', color=colors[1])
plt.title("Dataset 3:\n Train-1000-100 vs Test-1000-100\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Dataset 4
plt.plot(MSE_train_dataset_50_100, label='Train MSE', color=colors[0])
plt.plot(MSE_test_dataset_50_100, label='Test MSE', color=colors[1])
plt.title("Dataset 4:\n Train-50(1000)-100 vs Test-1000-100\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Dataset 5
plt.plot(MSE_train_dataset_100_100, label='Train MSE', color=colors[0])
plt.plot(MSE_test_dataset_100_100, label='Test MSE', color=colors[1])
plt.title("Dataset 5:\n Train-100(1000)-100 vs Test-1000-100\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Dataset 6
plt.plot(MSE_train_dataset_150_100, label='Train MSE', color=colors[0])
plt.plot(MSE_test_dataset_150_100, label='Test MSE', color=colors[1])
plt.title("Dataset 6:\n Train-150(1000)-100 vs Test-1000-100\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()

MSE_values = [MSE_test_dataset_1, MSE_test_dataset_2, MSE_test_dataset_3, MSE_test_dataset_50_100, MSE_test_dataset_100_100, MSE_test_dataset_150_100]

for i, MSE_value in enumerate(MSE_values, start=1):
    lambda_min = np.argmin(MSE_value)
    MSE_min = np.min(MSE_value)

    print(f"Dataset {i}: lambda = {lambda_min} gives the least MSE = {MSE_min}")


datasets = [
    (MSE_train_dataset_2_lambda_1_150, MSE_test_dataset_2_lambda_1_150, "Train-100-100 vs Test-100-100\n lambda= [1-150]"),
    (MSE_train_dataset_50_100_lambda_1_150, MSE_test_dataset_50_100_lambda_1_150, "Train-50(1000)-100 vs Test-1000-100\n lambda= [1-150]"),
    (MSE_train_dataset_100_100_lambda_1_150, MSE_test_dataset_100_100_lambda_1_150, "Train-100(1000)-100 vs Test-1000-100\n lambda= [1-150]")
]

for train_mse, test_mse, title in datasets:
    plt.plot(train_mse, label='Train MSE', color='purple')
    plt.plot(test_mse, label='Test MSE', color='pink')
    plt.title(title)
    plt.xlabel('Lambdas')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

KFolds = 10
# Find the fold size - Length of y matrix divided by number of folds
fold_size = int(len(y_matrix_train_1) / KFolds)

# MSE Sum
MSE_sum_test_1 = 0

# CV for Train-100-10, lambda: 0-150
for i in range(KFolds):
    X_test_fold = x_matrix_train_1[i * fold_size: (i + 1) * fold_size]
    Y_test_fold = y_matrix_train_1[i * fold_size: (i + 1) * fold_size]

    X_train_fold = np.concatenate((x_matrix_train_1[:i * fold_size], x_matrix_train_1[(i + 1) * fold_size:]), axis=0)
    Y_train_fold = np.concatenate((y_matrix_train_1[:i * fold_size], y_matrix_train_1[(i + 1) * fold_size:]), axis=0)

    weights = weight_calculation(X_train_fold, Y_train_fold, 150, lambda_start=0)

    MSE_sum_test_1 += mean_squared_error(X_test_fold, weights, Y_test_fold)

MSE_test_1 = MSE_sum_test_1 / KFolds

# Best lambda for the minimum test MSE value, 100-10, lambda: 0 - 150
lambda_min_test_1 = MSE_test_1.argmin()
MSE_min_test_1 = MSE_test_1[lambda_min_test_1]

print("Dataset 1 (100-10): lambda =", lambda_min_test_1, "gives the least test MSE =", MSE_min_test_1)

# Find the fold size - Length of y matrix divided by number of folds
fold_size_train_2 = int(len(y_matrix_train_2) / KFolds)

# MSE Sum
MSE_sum_test_2 = 0

# CV for Train-100-100, lambda: 0-150
for i in range(KFolds):
    X_test_2_fold = x_matrix_train_2[i * fold_size_train_2: (i + 1) * fold_size_train_2]
    Y_test_2_fold = y_matrix_train_2[i * fold_size_train_2: (i + 1) * fold_size_train_2]

    X_train_2_fold = np.concatenate((x_matrix_train_2[:i * fold_size_train_2], x_matrix_train_2[(i + 1) * fold_size_train_2:]), axis=0)
    Y_train_2_fold = np.concatenate((y_matrix_train_2[:i * fold_size_train_2], y_matrix_train_2[(i + 1) * fold_size_train_2:]), axis=0)

    weights_train_2 = weight_calculation(X_train_2_fold, Y_train_2_fold, 150, lambda_start=0)

    MSE_sum_test_2 += mean_squared_error(X_test_2_fold, weights_train_2, Y_test_2_fold)

MSE_test_2 = MSE_sum_test_2 / KFolds

# Best lambda for the minimum test MSE value, 100-100, lambda: 0 - 150
lambda_min_test_2 = MSE_test_2.argmin()
MSE_min_test_2 = MSE_test_2[lambda_min_test_2]

print("Dataset 2 (100-100): lambda =", lambda_min_test_2, "gives the least test MSE =", MSE_min_test_2)

# Find the fold size - Length of y matrix divided by the number of folds for Dataset 3
fold_size_train_3 = int(len(y_matrix_train_3) / KFolds)

# MSE Sum
MSE_sum_test_3 = 0

# CV for Train-1000-100, lambda: 0-150
for i in range(KFolds):
    X_test_3_fold = x_matrix_train_3[i * fold_size_train_3: (i + 1) * fold_size_train_3]
    Y_test_3_fold = y_matrix_train_3[i * fold_size_train_3: (i + 1) * fold_size_train_3]

    X_train_3_fold = np.concatenate((x_matrix_train_3[:i * fold_size_train_3], x_matrix_train_3[(i + 1) * fold_size_train_3:]), axis=0)
    Y_train_3_fold = np.concatenate((y_matrix_train_3[:i * fold_size_train_3], y_matrix_train_3[(i + 1) * fold_size_train_3:]), axis=0)

    weights_train_3 = weight_calculation(X_train_3_fold, Y_train_3_fold, 150, lambda_start=0)

    MSE_sum_test_3 += mean_squared_error(X_test_3_fold, weights_train_3, Y_test_3_fold)

MSE_test_3 = MSE_sum_test_3 / KFolds

# Best lambda for the minimum test MSE value, 1000-100, lambda: 0 - 150
lambda_min_test_3 = MSE_test_3.argmin()
MSE_min_test_3 = MSE_test_3[lambda_min_test_3]

print("Dataset 3 (1000-100): lambda =", lambda_min_test_3, "gives the least test MSE =", MSE_min_test_3)

# Find the fold size - Length of y matrix divided by the number of folds for Dataset 4
fold_size_train_50 = int(len(y_matrix_train_50) / KFolds)

# MSE Sum
MSE_sum_test_50 = 0

# CV for Train-50(1000-100), lambda: 0-150
for i in range(KFolds):
    X_test_50_fold = x_matrix_train_50[i * fold_size_train_50: (i + 1) * fold_size_train_50]
    Y_test_50_fold = y_matrix_train_50[i * fold_size_train_50: (i + 1) * fold_size_train_50]

    X_train_50_fold = np.concatenate((x_matrix_train_50[:i * fold_size_train_50], x_matrix_train_50[(i + 1) * fold_size_train_50:]), axis=0)
    Y_train_50_fold = np.concatenate((y_matrix_train_50[:i * fold_size_train_50], y_matrix_train_50[(i + 1) * fold_size_train_50:]), axis=0)

    weights_train_50 = weight_calculation(X_train_50_fold, Y_train_50_fold, 150, lambda_start=0)

    MSE_sum_test_50 += mean_squared_error(X_test_50_fold, weights_train_50, Y_test_50_fold)

MSE_test_50 = MSE_sum_test_50 / KFolds

# Best lambda for the minimum test MSE value, 50(1000)-100, lambda:0 - 150
lambda_min_test_50 = MSE_test_50.argmin()
MSE_min_test_50 = MSE_test_50[lambda_min_test_50]

print("Dataset 4 (50(1000)-100): lambda =", lambda_min_test_50, "gives the least test MSE =", MSE_min_test_50)

# Find the fold size - Length of y matrix divided by the number of folds for Dataset 5
fold_size_train_100 = int(len(y_matrix_train_100) / KFolds)

# MSE Sum
MSE_sum_test_100 = 0

# CV for Train-100(1000-100), lambda: 0-150
for i in range(KFolds):
    X_test_100_fold = x_matrix_train_100[i * fold_size_train_100: (i + 1) * fold_size_train_100]
    Y_test_100_fold = y_matrix_train_100[i * fold_size_train_100: (i + 1) * fold_size_train_100]

    X_train_100_fold = np.concatenate((x_matrix_train_100[:i * fold_size_train_100], x_matrix_train_100[(i + 1) * fold_size_train_100:]), axis=0)
    Y_train_100_fold = np.concatenate((y_matrix_train_100[:i * fold_size_train_100], y_matrix_train_100[(i + 1) * fold_size_train_100:]), axis=0)

    weights_train_100 = weight_calculation(X_train_100_fold, Y_train_100_fold, 150, lambda_start=0)

    MSE_sum_test_100 += mean_squared_error(X_test_100_fold, weights_train_100, Y_test_100_fold)

MSE_test_100 = MSE_sum_test_100 / KFolds

# Best lambda for the minimum test MSE value, 100(1000)-100, lambda:0 - 150
lambda_min_test_100 = MSE_test_100.argmin()
MSE_min_test_100 = MSE_test_100[lambda_min_test_100]

print("Dataset 5 (100(1000)-100): lambda =", lambda_min_test_100, "gives the least test MSE =", MSE_min_test_100)

# Find the fold size - Length of y matrix divided by the number of folds for Dataset 6
fold_size_train_150 = int(len(y_matrix_train_150) / KFolds)

# MSE Sum
MSE_sum_test_150 = 0

# CV for Train-150(1000-100), lambda: 0-150
for i in range(KFolds):
    X_test_150_fold = x_matrix_train_150[i * fold_size_train_150: (i + 1) * fold_size_train_150]
    Y_test_150_fold = y_matrix_train_150[i * fold_size_train_150: (i + 1) * fold_size_train_150]

    X_train_150_fold = np.concatenate((x_matrix_train_150[:i * fold_size_train_150], x_matrix_train_150[(i + 1) * fold_size_train_150:]), axis=0)
    Y_train_150_fold = np.concatenate((y_matrix_train_150[:i * fold_size_train_150], y_matrix_train_150[(i + 1) * fold_size_train_150:]), axis=0)

    weights_train_150 = weight_calculation(X_train_150_fold, Y_train_150_fold, 150, lambda_start=0)

    MSE_sum_test_150 += mean_squared_error(X_test_150_fold, weights_train_150, Y_test_150_fold)

MSE_test_150 = MSE_sum_test_150 / KFolds

# Best lambda for the minimum test MSE value, 150(1000)-100, lambda:0 - 150
lambda_min_test_150 = MSE_test_150.argmin()
MSE_min_test_150 = MSE_test_150[lambda_min_test_150]

print("Dataset 6 (150(1000)-100): lambda =", lambda_min_test_150, "gives the least test MSE =", MSE_min_test_150)

# Define function for learning curve
def learning_curve(x_matrix_train, y_matrix_train, x_matrix_test, y_matrix_test, rep, size):
    for lambda_value in [1, 25, 150]:
        size_list = range(10, 1000, size)
        MSE_array_test = np.zeros(len(size_list))
        MSE_array_train = np.zeros(len(size_list))
        for i in range(len(size_list)):
            rep_list_test = []
            rep_list_train = []

            for j in range(rep):
                idx = np.random.choice(len(x_matrix_train), size_list[i], replace=False)

                weights = weight_calculation(x_matrix_train[idx], y_matrix_train[idx], lambda_end=lambda_value, lambda_start=lambda_value)

                MSE_test = mean_squared_error(x_matrix_test, weights, y_matrix_test)

                MSE_train = mean_squared_error(x_matrix_train[idx], weights, y_matrix_train[idx])

                rep_list_test.append(MSE_test)
                rep_list_train.append(MSE_train)

            MSE_array_test[i] = np.average(rep_list_test)
            MSE_array_train[i] = np.average(rep_list_train)

        plt.plot(size_list, MSE_array_test, label='MSE_test', color = 'purple')
        plt.plot(size_list, MSE_array_train, label='MSE_train', color = 'pink')
        plt.xlabel('Training Set Size')
        plt.ylabel('MSE with lambda =' + str(lambda_value))
        plt.legend()
        plt.show()

# Call learning curve function for dataset 3 (1000-100)
learning_curve(x_matrix_train_3, y_matrix_train_3, x_matrix_test_3, y_matrix_test_3, 30, 10)
