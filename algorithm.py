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
        # drink_df = pd.DataFrame(self.data["Fun Drink"])
        # drink_df['EMA'] = drink_df[0].ewm(span=5, adjust=False).mean()
        # # Buy if the price is above the 5 day EMA
        # price_drink = self.data['Fun Drink'][-1]
        # ema = drink_df['EMA'].iloc[-1]
        # price_pens = self.data['Red Pens'][-1]  

        # theo = ema - 0.021*price_pens

        self.get_drink_position(desiredPositions, positionLimits)


        # if price_drink > theo:
        #     desiredPositions["Fun Drink"] = -positionLimits["Fun Drink"]
        # else:
        #     desiredPositions["Fun Drink"] = positionLimits["Fun Drink"]

        #######################################################################
        # Return the desired positions
        return desiredPositions
    
    def get_drink_position(self, desiredPositions, positionLimits):
        drinks_df = pd.DataFrame(self.data["Fun Drink"])
        pens_df = pd.DataFrame(self.data["Red Pens"])

        drinks_df['EMA'] = drinks_df[0].ewm(span=5, adjust=False).mean()
        pens_df['EMA'] = pens_df[0].ewm(span=5, adjust=False).mean()

        drinks_df['EMA25'] = drinks_df[0].ewm(span=25, adjust=False).mean()
        drinks_df['Cross'] = drinks_df['EMA'] - drinks_df['EMA25']

        price_drink = self.data['Fun Drink'][-1]
        price_pens = self.data['Red Pens'][-1]

        ema_drink = drinks_df['EMA'].iloc[-1]
        ema_pens = pens_df['EMA'].iloc[-1]
        
        cross_signal = drinks_df['Cross'].iloc[-1]

        theo = ema_drink -0.025*ema_pens + 0.055*np.sign(cross_signal)*(abs(cross_signal)**(1/4))

        if price_drink > theo:
            desiredPositions["Fun Drink"] = -positionLimits["Fun Drink"]
        else:
            desiredPositions["Fun Drink"] = positionLimits["Fun Drink"]