import numpy as np
import pandas as pd

# Custom trading Algorithm
class Algorithm_Sweeper():

    ########################################################
    # NO EDITS REQUIRED TO THESE FUNCTIONS
    ########################################################
    # FUNCTION TO SETUP ALGORITHM CLASS
    def __init__(self, positions, variable):
        # Initialise data stores:
        # Historical data of all instruments
        self.data = {}
        # Initialise position limits
        self.positionLimits = {}
        # Initialise the current day as 0
        self.day = 0
        # Initialise the current positions
        self.positions = positions
        self.var = variable

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
        # desiredPositions["UQ Dollar"] = self.get_uq_dollar_position(currentPositions["UQ Dollar"], positionLimits["UQ Dollar"])
        # Buy if the price is above the 5 day EMA
        # price_drink = self.data['Fun Drink'][-1]
        # ema = drinks_df['EMA'].iloc[-1]
        # price_pens = self.data['Red Pens'][-1]


        drinks_df = pd.DataFrame(self.data["Fun Drink"])
        pens_df = pd.DataFrame(self.data["Red Pens"])

        drinks_df['EMA'] = drinks_df[0].ewm(span=5, adjust=False).mean()
        pens_df['EMA'] = pens_df[0].ewm(span=5, adjust=False).mean()

        price_drink = self.data['Fun Drink'][-1]
        price_pens = self.data['Red Pens'][-1]

        ema_drink = drinks_df['EMA'].iloc[-1]
        ema_pens = pens_df['EMA'].iloc[-1]

        theo = ema_drink -0.025*ema_pens

        if price_drink > theo:
            desiredPositions["Fun Drink"] = -positionLimits["Fun Drink"]
        else:
            desiredPositions["Fun Drink"] = positionLimits["Fun Drink"]


        # drink_df = pd.DataFrame(self.data["Fun Drink"])
        # drink_df['EMA'] = drink_df[0].ewm(span=5, adjust=False).mean()
        # # Buy if the price is above the 5 day EMA
        # price = self.data['Fun Drink'][-1]
        # ema = drink_df['EMA'].iloc[-1]
        # if price > ema:
        #     desiredPositions["Fun Drink"] = -positionLimits["Fun Drink"]
        # else:
        #     desiredPositions["Fun Drink"] = positionLimits["Fun Drink"]

        # desiredPositions["Red Pens"] = self.get_red_pens_position(currentPositions["Red Pens"], positionLimits["Red Pens"])

        #######################################################################
        # Return the desired positions
        return desiredPositions
    
    def get_uq_dollar_position(self, currentPosition, limit):

        avg = sum(self.data["UQ Dollar"][-4:]) / 4
        price = self.get_current_price("UQ Dollar")
        diff = avg - price
        boundary = max(self.data["UQ Dollar"]) - avg

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

        return desiredPosition

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

        return desiredPosition
