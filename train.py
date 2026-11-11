import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

data_url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

def roast_category(roast):
    """Categorize roast levels into categories"""
    mapping = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }
    return mapping.get(roast, float('nan'))

def main():
    data = pd.read_csv(data_url)
    x = data[['100g_USD']]
    y = data['rating']

    model = LinearRegression()
    model.fit(x, y)

    with open("model_1.pickle", 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved as model_1.pickle")


    data["roast_cat"] = data["roast"].apply(roast_category)
    x1 = data[['100g_USD', 'roast_cat']]

    dtr = DecisionTreeRegressor(random_state=0)
    dtr.fit(x1, y)

    with open("model_2.pickle", 'wb') as f:
        pickle.dump(dtr, f)
    print("Model trained and saved as model_2.pickle")

if __name__ == "__main__":
    main()
