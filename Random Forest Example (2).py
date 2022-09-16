import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def random_forest_regression():
    # load dataset
    data = pd.read_csv(r"C:\Users\tomlm\Downloads\population.csv")
    # creating a dataframe of the given dataset

    # data = pd.DataFrame({
    #     'Population': data.values[:, 0],
    #     'Identifier': data.values[:, 1],
    #     'Year': data.values[:, 2],
    #     'World Population': data.values[:, 3]
    # })
    data.head()

    # x = data.iloc[0:, -2]  # features
    # y = data.iloc[0:, -1]  # labels

    X = data.iloc[:, 2:3].values
    y = data.iloc[:, 3].values

    # print(X)
    # print(y)


    # Fitting Random Forest Regression to the dataset
    # import the regressor
    from sklearn.ensemble import RandomForestRegressor
    # create regressor object
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)

    # fit the regressor with x and y data
    regressor.fit(X, y)

    Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))  # test the output by changing values

    # Visualising the Random Forest Regression results

    # arrange for creating a range of values
    # from min value of x to max
    # value of x with a difference of 0.01
    # between two consecutive values
    X_grid = np.arange(min(X), max(X), 0.01)

    # reshape for reshaping the data into a len(X_grid)*1 array,
    # i.e. to make a column out of the X_grid value
    X_grid = X_grid.reshape((len(X_grid), 1))

    # Scatter plot for original data
    plt.scatter(X, y, color='blue')

    # plot predicted data
    plt.plot(X_grid, regressor.predict(X_grid),
             color='green')
    plt.title('Random Forest Regression')
    plt.xlabel('Year')
    plt.ylabel('World Population')
    plt.show()
    print(regressor.predict(X_grid))




def vector_regression():
    # load dataset
    data = pd.read_csv(r"C:\Users\tomlm\Downloads\population.csv")
    data.head()

    X = data.iloc[:, 2:3].values
    # y = data.iloc[:, 3].values
    import csv
    with open(r"C:\Users\tomlm\Downloads\population.csv") as csvfile:
        reader = csv.reader(csvfile)
        table = [ row for row in reader ]
    del table[0]
    y = []
    for row in table:
        y = y + [[row[3]]]

    #3 Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X) #THIS LINE IS GIVING ME BRAIN HAEMORRHOIDS
    y = sc_y.fit_transform(y)

    #4 Fitting the Support Vector Regression Model to the dataset
    # Create your support vector regressor here
    from sklearn.svm import SVR
    # most important SVR parameter is Kernel type. It can be
    # #linear,polynomial or gaussian SVR. We have a non-linear condition
    # #so we can select polynomial or gaussian but here we select RBF(a
    # #gaussian type) kernel.
    regressor = SVR(kernel='rbf')
    regressor.fit(X,y)

    # #5 Predicting a new result
    # y_pred = regressor.predict(6.5)
    #
    # #5 Predicting a new result
    # y_pred = sc_y.inverse_transform ((regressor.predict (sc_X.transform(np.array([[6.5]])))))

    #6 Visualising the Support Vector Regression results
    plt.scatter(X, y, color = 'magenta')
    plt.plot(X, regressor.predict(X), color = 'green')
    plt.title('Truth or Bluff (Support Vector Regression Model)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


random_forest_regression()
vector_regression()

print()