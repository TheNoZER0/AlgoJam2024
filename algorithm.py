import numpy as np
from statsmodels.tsa.api import VAR, ARIMA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Custom trading Algorithm
class Algorithm():

    ########################################################
    # NO EDITS REQUIRED TO THESE FUNCTIONS
    ########################################################
    # FUNCTION TO SETUP ALGORITHM CLASS
    def __init__(self, positions):
        # Initialise data stores:
        self.data = {}
        # Initialise position limits
        self.positionLimits = {}
        # Initialise the current day as 0
        self.day = 0
        # Initialise the current positions
        self.positions = positions
        self.var_model = None
        self.scaler = StandardScaler()
        self.lookback = 1.1
        self.threshold = 0.0005  # Lowered threshold for increased sensitivity
        self.lag_order = 1
        self.var_instruments = ['Coffee Beans', 'Milk', 'Coffee']
        self.totalDailyBudget = 500000 

    # Helper function to fetch the current price of an instrument
    def get_current_price(self, instrument):
        return self.data[instrument][-1]
    ########################################################

    # HELPER METHODS

    # Regression model for Coffee
    def apply_regression_model(self, positionLimits, desiredPositions):
        if self.day >= self.lookback:
            # Get the most recent prices
            coffee_bean_price = self.data['Coffee Beans'][-1]
            milk_price = self.data['Milk'][-1]
            # Use the regression equation
            predicted_coffee_price = 1.5649 + 0.0077 * coffee_bean_price + 0.1565 * milk_price
            current_coffee_price = self.get_current_price('Coffee')
            # Decide on the position based on the predicted price
            price_diff = predicted_coffee_price - current_coffee_price
            if abs(price_diff) > self.threshold:
                # Use full position limit
                position = positionLimits['Coffee'] if price_diff > 0 else -positionLimits['Coffee']
                desiredPositions['Coffee'] = position

    # ARIMA model for trend-based instruments
    def apply_arima_model(self, instrument, positionLimits, desiredPositions, p=1, d=1, q=1):
        if self.day >= self.lookback:
            data = np.array(self.data[instrument])
            model = ARIMA(data, order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]

            current_price = self.get_current_price(instrument)
            price_diff = forecast - current_price
            if abs(price_diff) > self.threshold * current_price:
                # Use full position limit
                position = positionLimits[instrument] if price_diff > 0 else -positionLimits[instrument]
                desiredPositions[instrument] = position

    # Adjust positions to stay within budget
    def adjust_positions_for_budget(self, desiredPositions):
        total_value = 0
        prices = {inst: self.get_current_price(inst) for inst in desiredPositions}
        # Calculate total value of desired positions
        for inst, pos in desiredPositions.items():
            total_value += abs(pos * prices[inst])
        # If over budget, scale down positions proportionally
        if total_value > self.totalDailyBudget:
            scaling_factor = self.totalDailyBudget / total_value
            for inst in desiredPositions:
                desiredPositions[inst] = int(desiredPositions[inst] * scaling_factor)
        return desiredPositions

    # RETURN DESIRED POSITIONS IN DICT FORM
    def get_positions(self):
        # Get current position
        currentPositions = self.positions
        # Get position limits
        positionLimits = self.positionLimits

        # Declare a store for desired positions
        desiredPositions = {}
        # Initialise to hold positions for all instruments
        for instrument, positionLimit in positionLimits.items():
            desiredPositions[instrument] = 0

        # Apply Regression Model for Coffee
        self.apply_regression_model(positionLimits, desiredPositions)
        # IMPLEMENT CODE HERE TO DECIDE WHAT POSITIONS YOU WANT 
        #######################################################################
        # Buy thrifted jeans maximum amount
        desiredPositions["UQ Dollar"] = self.get_uq_dollar_position(currentPositions["UQ Dollar"], positionLimits["UQ Dollar"])
        
        jeans_df = pd.DataFrame(self.data["Thrifted Jeans"])
        jeans_df['EMA5'] = jeans_df[0].ewm(span=4, adjust=False).mean()
        # Buy if the price is above the 5 day EMA
        price = self.data['Thrifted Jeans'][-1]
        ema = jeans_df['EMA5'].iloc[-1]
        if price > ema:
            desiredPositions["Thrifted Jeans"] = -positionLimits["Thrifted Jeans"]
        else:
            desiredPositions["Thrifted Jeans"] = positionLimits["Thrifted Jeans"]

        # Apply ARIMA for Coffee Beans and Milk
        self.apply_arima_model("Coffee Beans", positionLimits, desiredPositions)
        self.apply_arima_model("Milk", positionLimits, desiredPositions)
        self.apply_arima_model("Fun Drink", positionLimits, desiredPositions)
        self.apply_arima_model("Red Pens", positionLimits, desiredPositions, p=1, d=1, q=0)
        # get_red_pens_position = self.get_red_pens_position(currentPositions["Red Pens"], positionLimits["Red Pens"])
        # desiredPositions["Red Pens"] = get_red_pens_position
        self.apply_arima_model("Red Pens", positionLimits, desiredPositions, p=1, d=1, q=0)
        self.apply_arima_model("Goober Eats", positionLimits, desiredPositions, p=1, d=1, q=1)
        self.apply_arima_model("Fintech Token", positionLimits, desiredPositions, p=2, d=1, q=2)

        # Adjust positions for budget
        desiredPositions = self.adjust_positions_for_budget(desiredPositions)

        return desiredPositions

    #######################################################################

    def get_uq_dollar_position(self, currentPosition, limit):
        avg = sum(self.data["UQ Dollar"][-4:]) / 4
        price = self.get_current_price("UQ Dollar")
        diff = avg - price
        boundary = max(self.data["UQ Dollar"]) - avg

        if diff > 0.15:
            delta = limit * 2
        elif diff < -0.15:
            delta = -2 * limit
        else:
            delta = 0

        if currentPosition + delta > limit:
            desiredPosition = limit
        elif currentPosition + delta < -1 * limit:
            desiredPosition = -1 * limit
        else:
            desiredPosition = currentPosition + delta

        return desiredPosition

    def get_red_pens_position(self, currentPosition, limit):
        avg = sum(self.data["Red Pens"][-10:]) / 10
        price = self.get_current_price("Red Pens")

        if price < 2.2:
            desiredPosition = limit
        elif price > 2.45:
            desiredPosition = -1 * limit
        else:
            desiredPosition = currentPosition

        return desiredPosition
