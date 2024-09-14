import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler

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

        #######################################################################
        # Return the desired positions
        return desiredPositions