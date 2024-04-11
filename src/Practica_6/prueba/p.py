import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# Step 1: Overfitting to training examples


def gen_data(m, seed=1, scale=0.7):
    c = 0
    x_train = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal * (np.random.sample((m,)) - 0.5)
    x_ideal = x_train
    return x_train, y_train, x_ideal, y_ideal


def overfitting(x_train, y_train, x_test, y_test):
    poly = PolynomialFeatures(degree=15, include_bias=False)
    x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
    x_test_poly = poly.transform(x_test.reshape(-1, 1))

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_poly)
    x_test_scaled = scaler.transform(x_test_poly)

    model = LinearRegression()
    model.fit(x_train_scaled, y_train)

    train_predictions = model.predict(x_train_scaled)
    test_predictions = model.predict(x_test_scaled)

    train_error = mean_squared_error(y_train, train_predictions)
    test_error = mean_squared_error(y_test, test_predictions)

    return train_error, test_error

# Step 2: Choosing the polynomial degree using a validation set


def select_polynomial_degree(x_train, y_train, x_val, y_val):
    degrees = np.arange(1, 11)
    train_errors = []
    val_errors = []

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
        x_val_poly = poly.transform(x_val.reshape(-1, 1))

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_poly)
        x_val_scaled = scaler.transform(x_val_poly)

        model = LinearRegression()
        model.fit(x_train_scaled, y_train)

        train_predictions = model.predict(x_train_scaled)
        val_predictions = model.predict(x_val_scaled)

        train_error = mean_squared_error(y_train, train_predictions)
        val_error = mean_squared_error(y_val, val_predictions)

        train_errors.append(train_error)
        val_errors.append(val_error)

    return degrees, train_errors, val_errors

# Step 3: Choosing the regularization parameter (λ)


def select_regularization_parameter(x_train, y_train, x_val, y_val):
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 600, 900]
    val_errors = []

    for l in lambdas:
        poly = PolynomialFeatures(degree=15, include_bias=False)
        x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
        x_val_poly = poly.transform(x_val.reshape(-1, 1))

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_poly)
        x_val_scaled = scaler.transform(x_val_poly)

        model = Ridge(alpha=l)
        model.fit(x_train_scaled, y_train)

        val_predictions = model.predict(x_val_scaled)
        val_error = mean_squared_error(y_val, val_predictions)

        val_errors.append(val_error)

    return lambdas, val_errors

# Step 4: Choosing hyperparameters


def select_hyperparameters(x_train, y_train, x_val, y_val):
    degrees = np.arange(1, 16)
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 600, 900]
    min_val_error = float('inf')
    best_degree = None
    best_lambda = None

    for degree in degrees:
        for l in lambdas:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
            x_val_poly = poly.transform(x_val.reshape(-1, 1))

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train_poly)
            x_val_scaled = scaler.transform(x_val_poly)

            model = Ridge(alpha=l)
            model.fit(x_train_scaled, y_train)

            val_predictions = model.predict(x_val_scaled)
            val_error = mean_squared_error(y_val, val_predictions)

            if val_error < min_val_error:
                min_val_error = val_error
                best_degree = degree
                best_lambda = l

    return best_degree, best_lambda

# Step 5: Learning curves


def learning_curves(x_train, y_train, x_val, y_val):
    num_examples = range(50, 651, 50)
    train_errors = []
    val_errors = []

    for num in num_examples:
        x_subset_train, _, y_subset_train, _ = train_test_split(
            x_train, y_train, train_size=num, random_state=1)

        poly = PolynomialFeatures(degree=16, include_bias=False)
        x_train_poly = poly.fit_transform(x_subset_train.reshape(-1, 1))
        x_val_poly = poly.transform(x_val.reshape(-1, 1))

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_poly)
        x_val_scaled = scaler.transform(x_val_poly)

        model = LinearRegression()
        model.fit(x_train_scaled, y_subset_train)

        train_predictions = model.predict(x_train_scaled)
        val_predictions = model.predict(x_val_scaled)

        train_error = mean_squared_error(y_subset_train, train_predictions)
        val_error = mean_squared_error(y_val, val_predictions)

        train_errors.append(train_error)
        val_errors.append(val_error)

    return num_examples, train_errors, val_errors


# Generate artificial data
np.random.seed(0)
x_train, y_train, _, _ = gen_data(64)
x_test, y_test, _, _ = gen_data(64)

# Split data into training, validation, and test sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=1)

# Step 1: Overfitting
train_error, test_error = overfitting(x_train, y_train, x_test, y_test)
print("Overfitting Train Error:", train_error)
print("Overfitting Test Error:", test_error)

# Step 2: Choosing polynomial degree
degrees, train_errors, val_errors = select_polynomial_degree(
    x_train, y_train, x_val, y_val)
best_degree = degrees[np.argmin(val_errors)]
print("Best Polynomial Degree:", best_degree)

# Step 3: Choosing regularization parameter
lambdas, val_errors = select_regularization_parameter(
    x_train, y_train, x_val, y_val)
best_lambda = lambdas[np.argmin(val_errors)]
print("Best Regularization Parameter:", best_lambda)

# Step 4: Choosing hyperparameters
best_degree, best_lambda = select_hyperparameters(
    x_train, y_train, x_val, y_val)
print("Best Polynomial Degree (Hyperparameter Selection):", best_degree)
print("Best Regularization Parameter (Hyperparameter Selection):", best_lambda)

# Step 5: Learning curves
num_examples, train_errors, val_errors = learning_curves(
    x_train, y_train, x_val, y_val)

# Plot learning curves
plt.plot(num_examples, train_errors, label='Train Error')
plt.plot(num_examples, val_errors, label='Validation Error')
plt.xlabel('Number of Training Examples')
plt.ylabel('Error')
plt.title('Learning Curves')
plt.legend()
plt.show()
