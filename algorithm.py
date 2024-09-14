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

        desiredPositions["Red Pens"] = self.get_red_pens_position(currentPositions["Red Pens"], positionLimits["Red Pens"])

        #######################################################################
        # Return the desired positions
        return desiredPositions
    
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
