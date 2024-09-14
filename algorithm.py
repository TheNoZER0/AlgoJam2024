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
        self.lookback = 15
        self.threshold = 0.01
        self.lag_order = 1
        self.var_instruments = ['Coffee Beans', 'Milk', 'Coffee']
        self.momentum_window = 5
        self.momentum_threshold = 0.02

    # Helper function to fetch the current price of an instrument
    def get_current_price(self, instrument):
        return self.data[instrument][-1]
    ########################################################

    # HELPER METHODS

    # VAR model for Coffee Beans, Milk, Coffee
    def apply_var_model(self, var_instruments, positionLimits, desiredPositions):
        if self.day >= self.lookback:
            var_data = np.array([self.data[inst] for inst in var_instruments if inst in self.data]).T
            if len(var_data) > self.lookback:
                diff_data = np.diff(var_data, axis=0) / var_data[:-1]
                scaled_data = self.scaler.fit_transform(diff_data)

                if self.var_model is None:
                    self.var_model = VAR(scaled_data)

                results = self.var_model.fit(self.lag_order)
                forecast = results.forecast(scaled_data[-1:], steps=1)[0]

                for i, instrument in enumerate(var_instruments):
                    if instrument in positionLimits:
                        if forecast[i] > self.threshold:
                            desiredPositions[instrument] = positionLimits[instrument]  # Buy
                        elif forecast[i] < -self.threshold:
                            desiredPositions[instrument] = -positionLimits[instrument]  # Sell

    # ARIMA model for trend-based instruments
    def apply_arima_model(self, instrument, positionLimits, desiredPositions, p=1, d=1, q=1):
        if self.day >= self.lookback:
            data = np.array(self.data[instrument])
            model = ARIMA(data, order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]

            current_price = self.get_current_price(instrument)
            if forecast > current_price * (1 + self.threshold):
                desiredPositions[instrument] = positionLimits[instrument]  # Buy
            elif forecast < current_price * (1 - self.threshold):
                desiredPositions[instrument] = -positionLimits[instrument]  # Sell

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

        # Apply VAR Model for Coffee Beans, Milk, Coffee
        self.apply_var_model(self.var_instruments, positionLimits, desiredPositions)

        # Apply ARIMA for trending instruments (Thrifted Jeans, Red Pens, Goober Eats, UQ Dollar, Fun Drink)
        self.apply_arima_model("Thrifted Jeans", positionLimits, desiredPositions)
        self.apply_arima_model("Red Pens", positionLimits, desiredPositions)
        self.apply_arima_model("Goober Eats", positionLimits, desiredPositions)
        self.apply_arima_model("UQ Dollar", positionLimits, desiredPositions)
        self.apply_arima_model("Fun Drink", positionLimits, desiredPositions)

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

        print(f"OLD: {currentPosition}, NEW: {desiredPosition}")
        
    def get_red_pens_position(self, currentPosition, limit):
        avg = sum(self.data["Red Pens"][-10:]) / 10
        price = self.get_current_price("Red Pens")

        if price < 2.2:
            desiredPosition = limit
        elif price > 2.45:
            desiredPosition = -1 * limit
        else:
            desiredPosition = currentPosition

        print(f"Old position: {currentPosition}, new position: {desiredPosition}")

        return desiredPosition
