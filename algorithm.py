import numpy as np
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
    # Helper function to fetch the current price of an instrument
    def get_current_price(self, instrument):
        # return most recent price
        return self.data[instrument][-1]
    ########################################################

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

        #######################################################################
        # Return the desired positions
        return desiredPositions