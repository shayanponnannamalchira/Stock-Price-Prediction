import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Initialize lists to store data
dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # Skip the header row (Date, Open, etc.)
        
        for row in csvFileReader:
            # Assuming Date is YYYY-MM-DD at index 0
            # We split by '-' and take the last part (the day)
            day = int(row[0].split('-')[2])
            dates.append(day)
            
            # Assuming the Price is at index 1
            prices.append(float(row[1]))
    return

def predict_prices(dates, prices, x):
    # Convert dates to an N x 1 matrix for sklearn
    dates = np.reshape(dates, (len(dates), 1))
    
    # Define the three SVR models mentioned in the video
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
    # Train (fit) the models on your NVIDIA data
    print("Training models...")
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    # Create the graph
    plt.scatter(dates, prices, color='black', label='Actual NVDA Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model (Best Fit)')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    
    plt.xlabel('Date (Day of Month)')
    plt.ylabel('Price ($)')
    plt.title('Support Vector Regression for NVIDIA')
    plt.legend()
    plt.show()
    
    # Return the predictions for the specific day 'x'
    return svr_rbf.predict(np.array([[x]]))[0], svr_lin.predict(np.array([[x]]))[0], svr_poly.predict(np.array([[x]]))[0]

# --- EXECUTION ---

# 1. Load your local file
get_data('nvidia.csv')

# 2. Predict for a future day (e.g., the 29th of the month)
rbf_pred, lin_pred, poly_pred = predict_prices(dates, prices, 29)

print(f"RBF Model Prediction: ${rbf_pred:.2f}")
print(f"Linear Model Prediction: ${lin_pred:.2f}")
print(f"Polynomial Model Prediction: ${poly_pred:.2f}")