import numpy as np
from statsmodels.tsa.api import VAR
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
        # Historical data of all instruments
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
        # return most recent price
        return self.data[instrument][-1]
    ########################################################

    # HELPER METHODS
    
    
    
    # RETURN DESIRED POSITIONS IN DICT FORM
    def get_positions(self):
        # Get current position
        currentPositions = self.positions
        # Get position limits
        positionLimits = self.positionLimits
        
        # Declare a store for desired positions
        desiredPositions = {}
        # Loop through all the instruments you can take positions on.
        for instrument, positionLimit in positionLimits.items():
            # For each instrument initilise desired position to zero
            desiredPositions[instrument] = 0

        # IMPLEMENT CODE HERE TO DECIDE WHAT POSITIONS YOU WANT 
        #######################################################################
        # Buy thrifted jeans maximum amount
        #desiredPositions["Thrifted Jeans"] = positionLimits["Thrifted Jeans"]
        
        # Coffee Milk and Beans VAR model
        if self.day >= self.lookback:
            var_data = np.array([self.data[inst] for inst in self.var_instruments if inst in self.data]).T
            if len(var_data) > self.lookback:
                diff_data = np.diff(var_data, axis=0) / var_data[:-1]
                scaled_data = self.scaler.fit_transform(diff_data)
                
                if self.var_model is None:
                    self.var_model = VAR(scaled_data)
                
                results = self.var_model.fit(self.lag_order)
                forecast = results.forecast(scaled_data[-1:], steps=1)[0]
                
                for i, instrument in enumerate(self.var_instruments):
                    if instrument in positionLimits:
                        if forecast[i] > self.threshold:
                            desiredPositions[instrument] = positionLimits[instrument]  # Buy
                        elif forecast[i] < -self.threshold:
                            desiredPositions[instrument] = -positionLimits[instrument]  # Sell
        
        #######################################################################
        # Other strategies
        desiredPositions["UQ Dollar"] = self.get_uq_dollar_position(currentPositions["UQ Dollar"], positionLimits["UQ Dollar"])
        
        desiredPositions["Thrifted Jeans"] = positionLimits["Thrifted Jeans"]
        jeans_df = pd.DataFrame(self.data["Thrifted Jeans"])
        jeans_df['EMA5'] = jeans_df[0].ewm(span=4, adjust=False).mean()
        # Buy if the price is above the 5 day EMA
        price = self.data['Thrifted Jeans'][-1]
        ema = jeans_df['EMA5'].iloc[-1]
        if price > ema:
            desiredPositions["Thrifted Jeans"] = -positionLimits["Thrifted Jeans"]
        else:
            desiredPositions["Thrifted Jeans"] = positionLimits["Thrifted Jeans"]

        desiredPositions["Red Pens"] = self.get_red_pens_position(currentPositions["Red Pens"], positionLimits["Red Pens"])

        #######################################################################
        # Return the desired positions
        return desiredPositions
    
    def get_uq_dollar_position(self, currentPosition, limit):

        avg = sum(self.data["UQ Dollar"][-4:]) / 4
        price = self.get_current_price("UQ Dollar")
        diff = avg - price
        boundary = max(self.data["UQ Dollar"]) - avg
        print(f"boundary: {boundary}")

        if diff > 0.15:
            delta = limit * 2 # int(np.exp(diff / boundary * 2) * limit)
        elif diff < -0.15:
            delta = -2 * limit # int(np.exp(abs(diff) / boundary * 2) * limit)
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

        # avg = self.penData.rolling(window=10, min_periods=1, on="Price").mean().at[self.day, "Price"]
        # price = self.penData.at[self.day, "Price"]
        avg = sum(self.data["Red Pens"][-10:]) / 10
        price = self.get_current_price("Red Pens")

        # Easy cases, these are great deals
        # if price < 2.19:
        #     desiredPosition = limit
        # elif price > 2.46:
        #     desiredPosition = -1 * limit
        # # Going up the slopes, if we for some reason haven't bought
        # elif avg > 2.23 and price > 2.24 and currentPosition < limit:
        #     desiredPosition = limit
        # # Going down the slopes, if we for some reason haven't sold
        # elif avg < 4.45 and price < 2.44 and currentPosition > -1 * limit:
        #     desiredPosition = -1 * limit
        # # If we're in the flat sections
        # elif avg < 2.23 and price < 2.2 and currentPosition < 0.9 * limit:
        #     desiredPosition = 0.95 * limit
        # elif avg < 2.23 and price > 2.21 and currentPosition > 0.9 * limit:
        #     desiredPosition = 0.85 * limit
        if price < 2.2:
            desiredPosition = limit
        elif price > 2.45:
            desiredPosition = -1 * limit
        else:
            desiredPosition = currentPosition

        print(f"Old position: {currentPosition}, new position: {desiredPosition}")

        return desiredPosition
