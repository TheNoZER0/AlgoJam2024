import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR, ARIMA
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

        self.lookback = 1.1
        self.threshold = 0.0002  # Lowered threshold for increased sensitivity
        self.lag_order = 1
        self.var_instruments = ['Coffee Beans', 'Milk', 'Coffee']
        self.totalDailyBudget = 500000 

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
        desiredPositions["UQ Dollar"] = self.get_uq_dollar_position(currentPositions["UQ Dollar"], positionLimits["UQ Dollar"])
        desiredPositions['Thrifted Jeans'] = self.get_jeans_position()
        desiredPositions["Red Pens"] = self.get_red_pens_position(currentPositions["Red Pens"], positionLimits["Red Pens"])
        desiredPositions["Goober Eats"] = self.get_goober_eats_position()
        desiredPositions["Fun Drink"] = self.get_fun_drink_position()
        desiredPositions["Fintech Token"] = self.get_token_position(currentPositions["Fintech Token"], positionLimits["Fintech Token"])
        
        # Apply Regression Model for Coffee
        self.apply_regression_model(positionLimits, desiredPositions)
        # Apply ARIMA for Coffee Beans and Milk
        self.apply_arima_model("Coffee Beans", positionLimits, desiredPositions)
        self.apply_arima_model("Milk", positionLimits, desiredPositions)

        desiredPositions = self.scale_positions(desiredPositions, currentPositions)


        #######################################################################
        # Return the desired positions
        return desiredPositions
    
    def scale_positions(self, desiredPositions, currentPositions):
        total_pos_value, prices_current, pos_values = self.calc_current_total_trade_val(desiredPositions, currentPositions)
        # if the total trade value is greater than the total daily budget, scale down the trade values for tokens
        if total_pos_value > self.totalDailyBudget:
            # find value we need to reduce by
            reduction_val = total_pos_value - self.totalDailyBudget
            # first reduce tokens because they are inneficient, but big size
            # find amount to reduce
            reduction_Tokens = int(reduction_val/prices_current['Fintech Token'])
            # if trades are positive, reduce trades, otherwise increase trades
            if pos_values['Fintech Token'] > 0:
                desiredPositions['Fintech Token'] -= min(reduction_Tokens, desiredPositions['Fintech Token'])
            else:
                desiredPositions['Fintech Token'] += min(reduction_Tokens, desiredPositions['Fintech Token'])               

            total_pos_value, prices_current, pos_values = self.calc_current_total_trade_val(desiredPositions, currentPositions)
            # if we are under the budget, return desired positions
            if total_pos_value <= self.totalDailyBudget:
                return desiredPositions

            # loop through the products in the order pens, dollar, beans, coffee, milk, goober, fun drink, jeans to get under
            for inst in ['Red Pens', 'UQ Dollar', 'Coffee Beans', 'Coffee', 'Milk', 'Goober Eats', 'Fun Drink', 'Thrifted Jeans']:
                # calculate required to reduce
                reduction_val = total_pos_value - self.totalDailyBudget
                # find amount to reduce
                reduction_inst = int(reduction_val/prices_current[inst])+1
                # reduce the desired positions
                if pos_values[inst] > 0:
                    desiredPositions[inst] -= min(reduction_inst, desiredPositions[inst])
                else:
                    desiredPositions[inst] += min(reduction_inst, -desiredPositions[inst])

                total_pos_value, prices_current, pos_values = self.calc_current_total_trade_val(desiredPositions, currentPositions)
                
                # if we are under the budget, break
                if total_pos_value <= self.totalDailyBudget:
                    return desiredPositions
        return desiredPositions
                
    def calc_current_total_trade_val(self, desiredPositions, currentPositions):
        # get prices for all instruments as a dictionary
        prices_current = {inst: self.get_current_price(inst) for inst in desiredPositions}
        # dict of trade values which is the trade amount multiplied by the current price
        pos_values = {inst: abs(desiredPositions[inst] * prices_current[inst]) for inst in desiredPositions}
        # total trade value is the sum of all trade values
        total_pos_value = sum(pos_values.values())
        # if self.day == 20:
        #     # stop all code execution and print all variables
        #     print('STOPPING')
        #     print('prices_current', prices_current)
        #     print('pos_values', pos_values)
        #     print('total_pos_value', total_pos_value)
        #     print('desiredPositions', desiredPositions)
        #     print('currentPositions', currentPositions)
        #     print('positionLimits', self.positionLimits)
        #     # stop code execution
        #     raise Exception('STOPPING')

        return total_pos_value, prices_current, pos_values

    def get_fun_drink_position(self):
        fun_df = pd.DataFrame(self.data["Fun Drink"])
        fun_df['EMA'] = fun_df[0].ewm(span=5, adjust=False).mean()
        # Buy if the price is above the 5 day EMA
        price = self.data['Fun Drink'][-1]
        ema = fun_df['EMA'].iloc[-1]
        limit = self.positionLimits["Fun Drink"]
        if price > ema:
            desiredPositions = -limit
        else:
            desiredPositions = limit
        return desiredPositions

    def get_goober_eats_position(self):
        goober_df = pd.DataFrame(self.data["Goober Eats"])
        goober_df['EMA'] = goober_df[0].ewm(span=13, adjust=False).mean()
        # Buy if the price is above the 5 day EMA
        price = self.data['Goober Eats'][-1]
        ema = goober_df['EMA'].iloc[-1]
        limit = self.positionLimits["Goober Eats"]
        if price > ema:
            desiredPositions = -limit
        else:
            desiredPositions = limit
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
    
    def get_jeans_position(self):
        jeans_df = pd.DataFrame(self.data["Thrifted Jeans"])
        jeans_df['EMA5'] = jeans_df[0].ewm(span=5, adjust=False).mean()
        # Buy if the price is above the 5 day EMA
        price = self.data['Thrifted Jeans'][-1]
        ema = jeans_df['EMA5'].iloc[-1]
        if price > ema:
            desiredPositions = -self.positionLimits["Thrifted Jeans"]
        else:
            desiredPositions = self.positionLimits["Thrifted Jeans"]
        return desiredPositions

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
    
    def get_token_position(self, currentPosition, limit):

        step = 35

        if self.day < 10:
            return currentPosition

        # Split the list into two halves
        first_half = self.data["Fintech Token"][-10:-5]
        second_half = self.data["Fintech Token"][-5:]

        # Calculate the gradients for each half
        first_grad = self.calculate_gradient(first_half)
        second_grad = self.calculate_gradient(second_half)

        lim = 18

        # Going from a stable section to jumping up
        if abs(first_grad) < lim and second_grad > lim:
            delta = step
        # Going from a stable section to jumping down
        elif abs(first_grad) < lim and second_grad < -1 * lim:
            delta = -1 * step
        else:
            delta = 0

        if currentPosition + delta > limit:
            desiredPosition = limit
        elif currentPosition + delta < -1 * limit:
            desiredPosition = -1 * limit
        else:
            desiredPosition = currentPosition + delta

        return desiredPosition
    

    # Function to calculate linear extrapolation
    def linear_extrapolation(self, values):
        if len(values) < 5:
            return np.nan  # Not enough data to extrapolate
        x = np.arange(5)
        y = values[-6:-1]
        coeffs = np.polyfit(x, y, 1)  # Linear fit (degree 1)
        extrapolated_value = np.polyval(coeffs, 5)  # Extrapolate to the next point
        return extrapolated_value
    
    def calculate_gradient(self, values):
        x = np.arange(5)  # Create an array [0, 1, 2, 3, 4] for 5 prices
        y = np.array(values)
        # Fit a linear model: y = mx + c
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        return m

    # # RETURN DESIRED POSITIONS IN DICT FORM
    # def get_positions(self):
    #     # Get current position
    #     currentPositions = self.positions
    #     # Get position limits
    #     positionLimits = self.positionLimits

    #     # Declare a store for desired positions
    #     desiredPositions = {}
    #     # Initialise to hold positions for all instruments
    #     for instrument, positionLimit in positionLimits.items():
    #         desiredPositions[instrument] = 0

    #     # Apply Regression Model for Coffee
    #     self.apply_regression_model(positionLimits, desiredPositions)

    #     # Apply ARIMA for Coffee Beans and Milk
    #     self.apply_arima_model("Coffee Beans", positionLimits, desiredPositions)
    #     self.apply_arima_model("Milk", positionLimits, desiredPositions)

    #     # Adjust positions for budget
    #     desiredPositions = self.adjust_positions_for_budget(desiredPositions)

    #     return desiredPositions