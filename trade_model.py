import pandas as pd
import numpy as np
import utility as utils


# ================================================================================================================
# Object that acts as a layer interface between trade model and main
# ================================================================================================================

class trade_model_to_main_interface:

    def __init__(self, initial_cash: float,
                 max_trade_val_percent: float, leverage: int,
                 trades_df: pd.DataFrame, last_unixtime_df: pd.DataFrame, timeframe_list: list, pair_price_dict: dict,
                 pair_min_trade_volume_dict: dict, pair_max_trade_price_decimal: dict, ohlc_csv_folder_name: str,
                 trade_csv_folder_name: str, trade_model: object):

        self.num_trail_stop_loss_update = 0
        self.trade_model = trade_model
        self.cash = initial_cash
        self.max_trade_val_percent = max_trade_val_percent
        self.max_trade_val = self.cash * max_trade_val_percent ### Max allowable $ per trade
        self.trade_df = trades_df
        self.last_unixtime_df = last_unixtime_df
        self.timeframe_list = timeframe_list
        self.pair_price_dict = pair_price_dict
        self.pair_min_trade_volume_dict = pair_min_trade_volume_dict
        self.pair_max_trade_price_decimal = pair_max_trade_price_decimal
        self.ohlc_csv_folder_name = ohlc_csv_folder_name
        self.trade_csv_folder_name = trade_csv_folder_name

        self.order_info_list = ['ordertype', 'type', 'pair', 'volume', 'price', 'price2', 'leverage',
                                'oflags', 'timeinforce', 'expiretm', 'close_ordertype', 'close_price',
                                'close_price2']

        self.leverage = leverage # Leverage (int). I.e., '2' = '2:1' leverage, '1' = '1:1' leverage, etc...

    def update_max_trade_val(self, cash):

        self.cash = cash
        self.max_trade_val = self.cash * self.max_trade_val_percent

    def check_for_existing_open_position(self, pair, trade_type, trigger):

        open_positions_df = utils.read_csv_data(self.trade_csv_folder_name, 'open_positions')
        account_balance_df = utils.read_csv_data(self.trade_csv_folder_name, 'account_balance')
        account_balance_df.rename(columns={'Unnamed: 0': 'ticker_index'}, inplace=True)
        account_balance_df.index = account_balance_df.ticker_index
        account_balance_df = account_balance_df.drop(columns=['ticker_index'])

        for row in range(open_positions_df.shape[0]):
            if open_positions_df.at[row, 'pair'] == pair and open_positions_df.at[row, 'type'] == trade_type:
                trigger = False

        if self.leverage == 1:
            ticker = utils.convert_pair_to_ticker(pair)

            if ticker in account_balance_df.ticker_corrected.to_list():

                ticker_row_vol = account_balance_df.vol[account_balance_df.ticker_corrected == ticker].item()

                if ticker_row_vol > 0 and trade_type == 'buy':
                    trigger = False

        return trigger

    def determine_final_trigger(self, pair_trigger_df: pd.DataFrame, pair: str):

        ### FUTURE - MODIFY TRIGGER TO IMPLEMENT 2 OUT OF 3 VOTING (I.E, IF TRIGGER = TRUE ON 2oo3 TIMEFRAMES, SEND TRADE)
        trade_type = None
        final_trigger = None

        sell_trade_trigger_count = 0
        buy_trade_trigger_count = 0

        print('1.0 T/S ------------------------')
        print(pair_trigger_df)

        for row in range(pair_trigger_df.shape[0]):

            trigger = pair_trigger_df.at[row, 'trigger']
            trade_type = pair_trigger_df.at[row, 'trade_type']

            if trigger == True and trade_type == 'buy':
                buy_trade_trigger_count += 1
            if trigger == True and trade_type == 'sell':
                sell_trade_trigger_count += 1

        if buy_trade_trigger_count >= 1:
            final_trigger = True
            trade_type = 'buy'
        if sell_trade_trigger_count >= 1:
            final_trigger = True
            trade_type = 'sell'

        final_trigger = self.check_for_existing_open_position(pair, trade_type, final_trigger)

        return final_trigger, trade_type

    def set_weight_of_max(self, ohlc_df_dict, trade_type):

        ### USED FOR OPENING NEW TRADE DIRECTION

        ### Initialize Weight of Max to 0. Variable will get set b/w 0 and 1 below pending trade type
        Weight_of_Max = 0

        if 240 in ohlc_df_dict:
            ohlc = ohlc_df_dict[240]
        else:
            last_key = list(ohlc_df_dict.keys())[-1]
            ohlc = ohlc_df_dict[last_key]

        current_price = ohlc.at[ohlc.shape[0]-1, 'close']
        BB_high = ohlc.at[ohlc.shape[0]-1, 'Bollinger_Band_High']
        BB_low = ohlc.at[ohlc.shape[0]-1, 'Bollinger_Band_Low']

        ### Use distance from BB lines as method for controlling volume of trade
        ### Ex - If price is closer to lower BB, increase trade volume percent weight of max allowable
        ### trade value.
        BB_distance = BB_high - BB_low

        if BB_low < current_price < BB_high:
            BB_diff = (current_price - BB_low)
        elif current_price > BB_high:
            BB_diff = BB_high - BB_low
        else:
            BB_diff = 0

        if trade_type == 'buy':

            ### Increase weight of max if price closer to lower BB range
            weight_of_max = 1 - (BB_diff / BB_distance)

            return weight_of_max

        elif trade_type == 'sell': ### For Trade_Type = Short

            ### Increaser weight of max if price closer to higher BB range
            weight_of_max = BB_diff / BB_distance

            return weight_of_max

    def set_trade_params(self, weight_of_max, ohlc_df_dict, pair):

        ### USED FOR OPENING NEW TRADE DIRECTION

        ohlc_df = list(ohlc_df_dict.values())[-1]
        max_price_decimal = self.pair_max_trade_price_decimal[pair]

        trade_book_val = weight_of_max * self.max_trade_val
        trade_unit_cost = ohlc_df.at[ohlc_df.shape[0]-1, 'close'].round(max_price_decimal)

        # Set to 4 decimal places max.
        trade_volume = (trade_book_val / trade_unit_cost)
        trade_volume = self.check_min_volume(pair, trade_volume)
        trade_volume = round(trade_volume, 4)

        # Recalculate trade book val based on rounded trade volume
        trade_book_val = trade_volume * trade_unit_cost

        return trade_book_val, trade_unit_cost, trade_volume

    def set_stop_loss(self, ohlc_df_dict, stop_trade_type, trade_volume, pair):

        max_price_decimal = self.pair_max_trade_price_decimal[pair]
        stop_price = self.trade_model.calc_stop_loss_unit_price(ohlc_df_dict= ohlc_df_dict, stop_trade_type= stop_trade_type)

        stop_price = round(stop_price, max_price_decimal)

        return stop_price, trade_volume

    def assign_new_trade_params(self, trade_type, trade_df_row,
                                trade_unit_cost, trade_volume,
                                stop_price):

        ### Parameters for new trade direction

        pair = self.trade_df.at[trade_df_row, 'pair']

        order_info_dict = {'ordertype': ["market"],
                           'type': [trade_type],
                           'pair': [pair],
                           'volume': [trade_volume],
                           'price': [trade_unit_cost],
                           'leverage': [str(self.leverage) + ':1'],
                           'close_ordertype': ["stop-loss"],
                           'close_price': [stop_price]}

        order_info_df = pd.DataFrame(data= order_info_dict)

        return order_info_df

    def assign_close_trade_params(self, close_trade_type, trade_df_row, unit_cost):

        ### Parameters to close existing open trade
        order_info_df = pd.DataFrame(columns= ['ordertype', 'type', 'pair', 'volume', 'price', 'leverage'])

        open_positions_df = utils.read_csv_data(csv_folder_name = self.trade_csv_folder_name, csv_file_name= 'open_positions',
                                                LimitPrevNumRows= None).drop(columns= ['Unnamed: 0'])

        account_balance_df = utils.read_csv_data(csv_folder_name = self.trade_csv_folder_name, csv_file_name= 'account_balance',
                                                 LimitPrevNumRows= None)
        account_balance_df.rename(columns={'Unnamed: 0': 'ticker'}, inplace=True)
        account_balance_df.index = account_balance_df.ticker

        pair = self.trade_df.at[trade_df_row, 'pair']
        aggregate_offside_volume = 0 ### INITIALIZE VARIABLE FOR SUM OF VOLUME THAT IS ON WRONG SIDE OF CURRENT TRIGGER SIGNAL

        for row in range(open_positions_df.shape[0]):
            if open_positions_df.at[row, 'pair'] == pair and open_positions_df.at[row, 'type'] != close_trade_type:
                aggregate_offside_volume += (open_positions_df.at[row, 'vol'] -
                                             open_positions_df.at[row, 'vol_closed'])

        # For a spot only strategy, find existing spot position to close, if required. (Leverage = 1 for spot only)
        if close_trade_type == 'sell' and self.leverage == 1:
            bal_acc_ticker = pair.replace('USD', '')
            if bal_acc_ticker in account_balance_df.index.to_list():
                aggregate_offside_volume += account_balance_df.at[bal_acc_ticker, 'vol']

        if aggregate_offside_volume > 0:

            order_info_dict = {'ordertype': ["market"],
                               'type': [close_trade_type],
                               'pair': [self.trade_df.at[trade_df_row, 'pair']],
                               'volume': [aggregate_offside_volume],
                               'price': [unit_cost],
                               'leverage': [str(self.leverage) + ':1']}

            order_info_df = pd.DataFrame(data= order_info_dict)

        return order_info_df

    def is_trade(self):

        ### LOOP FUCTION OVER OHLC DATA. IF TRIGGER = TRUE, THEN RETURN TRADE PARAMETERS TO PLUG INTO PYKRAKENAPI
        ### "add_standard_order" FUNCTION

        new_order_info_df = pd.DataFrame(columns= self.order_info_list)
        close_order_info_df = pd.DataFrame(columns= self.order_info_list)
        all_ohlc_df_dict = {}

        for row_pair in range(self.trade_df.shape[0]):
            ohlc_df_dict = {}
            pair = self.trade_df.at[row_pair, 'pair']

            # Read all OHLC data for this pair, for all timeframes
            for time in self.timeframe_list:
                ohlc_filename = pair + '_' + str(time)
                ohlc = utils.read_csv_data(csv_folder_name=self.ohlc_csv_folder_name, csv_file_name=ohlc_filename,
                                           LimitPrevNumRows=20)
                ohlc_df_dict[time] = ohlc

            all_ohlc_df_dict[pair] = ohlc_df_dict

            pair_trigger_df = self.trade_model.create_trigger_df(ohlc_df_dict= ohlc_df_dict, pair= pair)
            trigger, trade_type = self.determine_final_trigger(pair_trigger_df, pair)
            weight_of_max = self.set_weight_of_max(ohlc_df_dict = ohlc_df_dict, trade_type= trade_type) # Using the four hour for BB

            if weight_of_max == 0:
                trigger = False

            if trigger == True:


                trade_book_val, trade_unit_cost, trade_volume = self.set_trade_params(weight_of_max= weight_of_max, ohlc_df_dict= ohlc_df_dict, pair= pair)

                print('2.0 ORDER PRICE T/S')
                print(trade_book_val, trade_unit_cost, trade_volume)

                if trade_type == 'buy':
                    stop_trade_type = 'sell'
                else:
                    stop_trade_type = 'buy'

                stop_price, stop_volume = self.set_stop_loss(ohlc_df_dict, stop_trade_type, trade_volume, pair)

                ### Output the close order dict if existing offside position exists. This function checks for
                ### existing position. Will return dict of None if no existing open position.
                close_order_info_df_temp = self.assign_close_trade_params(trade_type, row_pair, trade_unit_cost)

                ### NEW TRADE PARAMS
                new_order_info_df_temp = self.assign_new_trade_params(trade_type, row_pair, trade_unit_cost,
                                                                      trade_volume, stop_price)

            else:

                close_order_info_df_temp = pd.DataFrame(columns= self.order_info_list)
                new_order_info_df_temp = pd.DataFrame(columns= self.order_info_list)


            if new_order_info_df_temp.shape[0] != 0:
                new_order_info_df = pd.concat([new_order_info_df, new_order_info_df_temp], ignore_index= True)
            if close_order_info_df_temp.shape[0] != 0:
                close_order_info_df = pd.concat([close_order_info_df, close_order_info_df_temp], ignore_index= True)

        # Cancel open orders that are now offside/obsolete
        cancel_orders_df = self.cancel_existing_obsolete_open_orders(close_order_info_df)

        # If using spot only, filter out margin short orders
        if self.leverage == 1:
            new_order_info_df_temp = new_order_info_df
            for row in range(new_order_info_df.shape[0]):
                if new_order_info_df.at[row, 'type'] == 'sell':
                    new_order_info_df_temp = new_order_info_df_temp.drop(row)
            new_order_info_df = new_order_info_df_temp.reset_index(drop= True)

        updated_orders_df, cancel_trail_orders_df = self.update_existing_stop_loss(all_ohlc_df_dict, cancel_orders_df)

        return new_order_info_df, close_order_info_df, cancel_orders_df, updated_orders_df, cancel_trail_orders_df

    def cancel_existing_obsolete_open_orders(self, order_info_df):

        '''

        :param order_info_df: dataframe of newly triggered orders from trade model
        :return:
            cancel_orders_df: dataframe of orders to cancel. These are orders that are now offsides relative
            to newly triggered direction (generally stop-losses)
        '''

        open_orders_df = utils.read_csv_data(csv_folder_name = self.trade_csv_folder_name,
                                             csv_file_name= 'open_orders',
                                             LimitPrevNumRows= None)

        open_orders_df.rename(columns={'Unnamed: 0': 'index_col'}, inplace=True)

        cancel_orders_df = pd.DataFrame(columns= ['ordertxid'])

        for row_new_order in range(order_info_df.shape[0]):

            for row_open_order in range(open_orders_df.shape[0]):
                if open_orders_df.at[row_open_order, 'descr_pair'] == order_info_df.at[row_new_order, 'pair'] and \
                        open_orders_df.at[row_open_order, 'descr_type'] == order_info_df.at[row_new_order, 'type']:

                    cancel_orders_df.at[cancel_orders_df.shape[0], 'ordertxid'] = open_orders_df.at[row_open_order, 'index_col']

        return cancel_orders_df

    def update_existing_stop_loss(self, all_ohlc_df_dict: dict, cancel_existing_orders_df: pd.DataFrame):

        '''
        Update existing stop-loss positions for trailing stop loss strategy

        :return:
        '''

        open_orders_df = utils.read_csv_data(self.trade_csv_folder_name, 'open_orders').reset_index()
        open_orders_df.rename(columns={'Unnamed: 0': 'ordertxid'}, inplace=True)
        cancel_orders_df = pd.DataFrame(columns= ['ordertxid'])
        updated_orders_df = pd.DataFrame(columns= self.order_info_list)

        for row_open_orders in range(open_orders_df.shape[0]):
            if open_orders_df.at[row_open_orders, 'ordertxid'] in cancel_existing_orders_df.ordertxid.to_list():
                open_orders_df = open_orders_df.drop(row_open_orders)
        open_orders_df = open_orders_df.reset_index(drop= True)

        for row in range(open_orders_df.shape[0]):

            pair = open_orders_df.at[row, 'descr_pair']
            current_pair_price = self.pair_price_dict[pair]
            current_order_price = open_orders_df.at[row, 'descr_price']
            stop_trade_type = open_orders_df.at[row, 'descr_type']
            volume = open_orders_df.at[row, 'vol'] - open_orders_df.at[row, 'vol_exec']
            pair_max_trade_price_decimal = self.pair_max_trade_price_decimal[pair]
            ohlc_df_dict = all_ohlc_df_dict[pair]

            new_stop_loss_price, is_update_order = self.trade_model.calc_trail_stop_loss(ohlc_df_dict, stop_trade_type, current_pair_price, current_order_price)

            if is_update_order == True and new_stop_loss_price != current_order_price:
                new_stop_loss_price = round(new_stop_loss_price, pair_max_trade_price_decimal)

                index = updated_orders_df.shape[0]

                cancel_orders_df.at[index, 'ordertxid'] = open_orders_df.at[row, 'ordertxid']

                updated_orders_df.at[index, 'ordertype'] = 'stop-loss'
                updated_orders_df.at[index, 'type'] = stop_trade_type
                updated_orders_df.at[index, 'pair'] = pair
                updated_orders_df.at[index, 'volume'] = volume
                updated_orders_df.at[index, 'price'] = new_stop_loss_price
                updated_orders_df.at[index, 'leverage'] = str(self.leverage) + ':1'

                self.num_trail_stop_loss_update += 1

                print('1.0 UPDATE TRAILING STOP LOSS TRIGGERED')
                print(cancel_orders_df)
                print('1.1 UPDATE TRAILING STOP LOSS TRIGGERED')
                print(updated_orders_df)
                print('1.2 UPDATE TRAILING STOP LOSS TRIGGERED')
                print(self.num_trail_stop_loss_update)

        return updated_orders_df, cancel_orders_df

    def check_min_volume(self, pair, volume):

        trade_val = self.pair_price_dict[pair] * volume

        if volume < self.pair_min_trade_volume_dict[pair]:
            volume = self.pair_min_trade_volume_dict[pair]
            trade_val = self.pair_price_dict[pair] * volume
        if trade_val > self.cash:
            volume = 0

        return volume

# ================================================================================================================
# TRADE MODELS
# ================================================================================================================
class stoch_rsi_model:

    stoch_RSI_high_threshold = 80
    stoch_RSI_low_threshold = 20

    def create_trigger_df(self, pair: str, ohlc_df_dict: dict):

        trigger_df = pd.DataFrame(columns= ['pair', 'trade_type', 'trigger'])

        ohlc_df_list = list(ohlc_df_dict.values())

        for row in range(len(ohlc_df_list)):

            stoch_rsi_trigger_row_df_temp = self._determine_stoch_rsi_trigger(ohlc_df= ohlc_df_list[row], pair= pair)
            trigger_df = pd.concat([trigger_df, stoch_rsi_trigger_row_df_temp], ignore_index=True)

        trigger_df = trigger_df.reset_index(drop= True)

        return trigger_df

    def _determine_stoch_rsi_trigger(self, ohlc_df: pd.DataFrame, pair: str):

        '''

        :param pair: (str) Ticker/USD pair
        :param timeframe: Candle time frame (i.e., 1 Hr, 4 Hr, 1WK, etc)
        :return:
        '''

        # Initialize variables
        trade_type = None
        trigger = False
        prime_high = False
        prime_low = False

        stoch_rsi_prev = float(ohlc_df.at[ohlc_df.shape[0]-2, 'Stoch RSI'])
        stoch_rsi_curr = float(ohlc_df.at[ohlc_df.shape[0]-1, 'Stoch RSI'])
        stoch_rsi_trigger_df = pd.DataFrame()

        if stoch_rsi_prev > self.stoch_RSI_high_threshold:
            prime_high = True
        elif stoch_rsi_prev < self.stoch_RSI_low_threshold:
            prime_low = True

        if prime_high == True and stoch_rsi_curr < self.stoch_RSI_high_threshold:
            trigger = True
            trade_type = 'sell'

        elif prime_low == True and stoch_rsi_curr > self.stoch_RSI_low_threshold:
            trigger = True
            trade_type = 'buy'

        stoch_rsi_trigger_df.at[0, 'pair'] = pair
        stoch_rsi_trigger_df.at[0, 'trade_type'] = trade_type
        stoch_rsi_trigger_df.at[0, 'trigger'] = trigger

        print('4.0 determine trigger ts ------------------------')
        print(pair)
        print('4.1 determine trigger ts ------------------------')
        print(ohlc_df.at[ohlc_df.shape[0]-1, 'time'])
        print('4.2 determine trigger ts ------------------------')
        print(stoch_rsi_prev, stoch_rsi_curr)
        print('4.3 determine trigger ts ------------------------')
        print(trigger)
        print('4.4 determine trigger ts ------------------------')
        print(trade_type)


        return stoch_rsi_trigger_df

class EMA_cross_model:

    def create_trigger_df(self, pair: str, ohlc_df_dict: dict):

        trigger_df = pd.DataFrame(columns= ['pair', 'trade_type', 'trigger'])

        ohlc_df_list = list(ohlc_df_dict.values())

        for row in range(len(ohlc_df_list)):

            stoch_rsi_trigger_row_df_temp = self._EMA_cross_trigger(ohlc_df= ohlc_df_list[row], pair= pair)
            trigger_df = pd.concat([trigger_df, stoch_rsi_trigger_row_df_temp], ignore_index=True)

        trigger_df = trigger_df.reset_index(drop= True)

        return trigger_df

    def _EMA_cross_trigger(self, ohlc_df: pd.DataFrame, pair: str):

        # Initialize variables
        trade_type = None
        trigger = False
        EMA_cross_trigger_df = pd.DataFrame()

        EMA_12 = float(ohlc_df.at[ohlc_df.shape[0]-1, 'EMA_12'])
        EMA_21 = float(ohlc_df.at[ohlc_df.shape[0]-1, 'EMA_21'])

        if EMA_12 > EMA_21:
            trigger = True
            trade_type = 'buy'
        elif EMA_12 < EMA_21:
            trigger = True
            trade_type = 'sell'

        EMA_cross_trigger_df.at[0, 'pair'] = pair
        EMA_cross_trigger_df.at[0, 'trade_type'] = trade_type
        EMA_cross_trigger_df.at[0, 'trigger'] = trigger


        print('1.0 EMA_cross_trigger ------------------------------------------------')
        print_df = pd.DataFrame(data= {'pair': [pair],
                                       'prev_time': [ohlc_df.at[ohlc_df.shape[0] - 2, 'dtime']],
                                       'curr_time': [ohlc_df.at[ohlc_df.shape[0] - 1, 'dtime']],
                                       'prev_unix': [ohlc_df.at[ohlc_df.shape[0] - 2, 'time']],
                                       'curr_unix': [ohlc_df.at[ohlc_df.shape[0] - 1, 'time']],
                                       'EMA_12': [EMA_12],
                                       'EMA_21': [EMA_21],
                                       'trigger': [trigger],
                                       'trade_type': [trade_type]})
        print(print_df)

        return EMA_cross_trigger_df

class EMA_cross_model_cross_timeframe_voting:

    def __init__(self, EMA_cross_model: object):

        self.EMA_cross_model = EMA_cross_model

    def create_trigger_df(self, pair: str, ohlc_df_dict: dict):

        EMA_trigger_df = self.EMA_cross_model.create_trigger_df(pair, ohlc_df_dict)
        EMA_trigger_df_temp = EMA_trigger_df[0:EMA_trigger_df.shape[0]-1] #include up to 2nd last

        high_timeframe_momentum = EMA_trigger_df.at[EMA_trigger_df.shape[0]-1, 'trade_type']
        onside_counter = EMA_trigger_df_temp[EMA_trigger_df_temp.trade_type == high_timeframe_momentum].shape[0]
        offside_counter = EMA_trigger_df_temp.shape[0] - onside_counter

        trigger_df = pd.DataFrame(columns= ['pair', 'trade_type', 'trigger'])

        # If most of the indicators are same side then trigger = true

        if onside_counter >= EMA_trigger_df_temp.shape[0] - 1:
            trade_type = high_timeframe_momentum
            trigger = True
        elif offside_counter == EMA_trigger_df_temp.shape[0]:
            trigger = True
            if high_timeframe_momentum == 'buy':
                offside_trade_type = 'sell'
            else:
                offside_trade_type = 'buy'
            trade_type = offside_trade_type
        else:
            trade_type = None
            trigger = False

        trigger_df.at[0, 'pair'] = pair
        trigger_df.at[0, 'trade_type'] = trade_type
        trigger_df.at[0, 'trigger'] = trigger

        return trigger_df

    def calc_stop_loss_unit_price(self, ohlc_df_dict: dict, stop_trade_type: str):

        if 240 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[240]
        else:
            ohlc_df = list(ohlc_df_dict.values())[0]

        # stop_price = calc_chandelier_exit_stop(ohlc_df= ohlc_df, stop_trade_type= stop_trade_type)

        max_trade_loss_percent = 0.10
        stop_price = calc_simple_stop(ohlc_df= ohlc_df , stop_trade_type= stop_trade_type,max_trade_loss_percent= max_trade_loss_percent)

        return stop_price

    def calc_trail_stop_loss(self, ohlc_df_dict: dict, stop_trade_type, current_pair_price, current_order_price):

        trailing_stop_loss_percent_threshold= 0.15
        trailing_stop_loss_percent_change = 0.08

        new_stop_loss_price, is_update_order = calc_updated_simple_trail_stop(stop_trade_type,
                                                                              current_pair_price,
                                                                              current_order_price,
                                                                              trailing_stop_loss_percent_threshold,
                                                                              trailing_stop_loss_percent_change)

        return new_stop_loss_price, is_update_order

class support_resistance_model:

    def find_sr_range(self, pair: str, timeframe: int= 1440):

        sr_lines = utils.find_sr_lines(pair, timeframe, num_data_points= 60)

        fib_0 = sr_lines[0][0]
        fib_100 = sr_lines[-1][0]
        fib_50 = fib_0 + ((fib_100 - fib_0) / 2)

        return fib_0, fib_50, fib_100

    def _sr_trigger(self, pair, current_price, timeframe: int= 1440):

        fib_0, fib_50, fib_100 = self.find_sr_range(pair= pair, timeframe= timeframe, num_data_points= 60)
        sr_trigger_df = pd.DataFrame()

        if current_price <= fib_0:
            trade_type = 'buy'
            trigger = True
        elif current_price >= fib_100:
            trade_type = 'sell'
            trigger = True
        else:
            trade_type = None
            trigger = False

        sr_trigger_df.at[0, 'pair'] = pair
        sr_trigger_df.at[0, 'trade_type'] = trade_type
        sr_trigger_df.at[0, 'trigger'] = trigger

        print('1.0 S/R TRIGGER DF ---------------------------------------')
        print(sr_trigger_df)
        print(fib_0, fib_50, fib_100)

        return sr_trigger_df

    def create_trigger_df(self, pair: str, ohlc_df_dict: dict):

        trigger_df = pd.DataFrame(columns= ['pair', 'trade_type', 'trigger'])

        if 1440 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[1440]
        else:
            ohlc_df_list = list(ohlc_df_dict.values())
            ohlc_df = ohlc_df_list[-1]

        current_price = ohlc_df.at[ohlc_df.shape[0]-1, 'close']

        sr_trigger_row_df_temp = self._sr_trigger(pair= pair, current_price= current_price, timeframe= 1440)
        trigger_df = pd.concat([trigger_df, sr_trigger_row_df_temp], ignore_index=True)

        trigger_df = trigger_df.reset_index(drop= True)

        return trigger_df

    def calc_stop_loss_unit_price(self, ohlc_df_dict: dict, stop_trade_type: str):

        if 60 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[60]
        else:
            ohlc_df = list(ohlc_df_dict.values())[-1]

        stop_price = calc_chandelier_exit_stop(ohlc_df=ohlc_df, stop_trade_type=stop_trade_type)

        return stop_price

    def calc_trail_stop_loss(self, ohlc_df_dict: dict, stop_trade_type, current_pair_price, current_order_price):

        trailing_stop_loss_percent_threshold = 0.08
        trailing_stop_loss_percent_change = 0.04

        new_stop_loss_price, is_update_order = calc_updated_simple_trail_stop(stop_trade_type,
                                                                              current_pair_price,
                                                                              current_order_price,
                                                                              trailing_stop_loss_percent_threshold,
                                                                              trailing_stop_loss_percent_change)

        return new_stop_loss_price, is_update_order

class bollinger_derivative_reversal_model:

    def __init__(self, pairs: list):

        self.pairs = pairs

        self.trigger_matrix = pd.DataFrame(data={
            'pairs': pairs,
            'bb_trade_type': [None] * len(pairs),
            'bb_trigger': [False] * len(pairs),
            'ema_deriv_trade_type': [None] * len(pairs),
            'ema_deriv_trigger': [False] * len(pairs)
        })

    def _check_bb_trigger(self, ohlc_df: pd.DataFrame, pair: str):

        '''
        Check if price had touched upper or lower bollinger band. Check the last two rows of ohlc df.

        Checking last two rows will help ensure against missing a data point that has low or a high close above/below BB.
        Particularly important for test mode because it increments forward in large chunks of time (say 30 mins).

        :param ohlc_df:
        :param pair:
        :return:
        '''

        last_row_index = ohlc_df.shape[0] - 1
        last_two_rows = ohlc_df.loc[last_row_index - 1: last_row_index + 1, :]

        trade_type = self.trigger_matrix[self.trigger_matrix.pairs == pair].bb_trade_type.to_list()[0]
        trigger = self.trigger_matrix[self.trigger_matrix.pairs == pair].bb_trigger.to_list()[0]

        current_price = ohlc_df.at[last_row_index, 'close']


        is_outside_bb_range = np.where(np.logical_and(last_two_rows.high > last_two_rows.Bollinger_Band_High,
                                                      last_two_rows.low < last_two_rows.Bollinger_Band_Low),
                                       True, np.nan)
        is_touch_bb_low = np.where(last_two_rows.low <= last_two_rows.Bollinger_Band_Low, True, np.nan)
        is_touch_bb_high = np.where(last_two_rows.high >= last_two_rows.Bollinger_Band_High, True, np.nan)

        if True in is_outside_bb_range:
            trade_type = None
            trigger = False

        elif True in is_touch_bb_high:
            trade_type = 'sell'
            trigger = True

        elif True in is_touch_bb_low:
            trade_type = 'buy'
            trigger = True

        return trade_type, trigger

    def _check_ema_derivative_trigger(self, ohlc_df: pd.DataFrame, pair: str):

        # EMA_array = ohlc_df.EMA_12
        # EMA_derivative_array = utils.calc_derivative_array(EMA_array)

        EMA_derivative_array = ohlc_df.ema_12_derivative.to_list()

        current_derivative = EMA_derivative_array[len(EMA_derivative_array) - 1]

        if current_derivative > 0:
            trade_type = 'buy'
            trigger = True
        elif current_derivative < 0:
            trade_type = 'sell'
            trigger = True
        else:
            trade_type = None
            trigger = False

        return trade_type, trigger

    def create_trigger_df(self, pair: str, ohlc_df_dict: dict):

        trigger_df = pd.DataFrame(data = {
            'pair': [pair],
            'trade_type': [None],
            'trigger': [False]
        })

        if 60 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[60]
        else:
            ohlc_df_list = list(ohlc_df_dict.values())
            ohlc_df = ohlc_df_list[0]

        bb_trade_type, bb_trigger = self._check_bb_trigger(pair= pair, ohlc_df= ohlc_df)
        ema_derivative_trade_type, ema_derivative_trigger = self._check_ema_derivative_trigger(pair= pair, ohlc_df= ohlc_df)

        pair_ind = self.trigger_matrix[self.trigger_matrix.pairs == pair].index[0]
        self.trigger_matrix.at[pair_ind, 'bb_trade_type'] = bb_trade_type
        self.trigger_matrix.at[pair_ind, 'bb_trigger'] = bb_trigger
        self.trigger_matrix.at[pair_ind, 'ema_deriv_trade_type'] = ema_derivative_trade_type
        self.trigger_matrix.at[pair_ind, 'ema_deriv_trigger'] = ema_derivative_trigger

        print('1.0 TRADE MODE - TRIGGER DF ---------------------------------------------' )
        print(pair)
        print(self.trigger_matrix)
        print(ohlc_df.loc[ohlc_df.shape[0]-2:ohlc_df.shape[0]-1, :])
        print('\n')

        if bb_trigger and ema_derivative_trigger == True:
            trigger = bb_trigger
            trade_type = None
            if bb_trade_type == ema_derivative_trade_type:
                trade_type = bb_trade_type
        else:
            trade_type = None
            trigger = False

        trigger_df.at[0, 'pair'] = pair
        trigger_df.at[0, 'trade_type'] = trade_type
        trigger_df.at[0, 'trigger'] = trigger

        return trigger_df

    def calc_stop_loss_unit_price(self, ohlc_df_dict: dict, stop_trade_type: str):

        if 60 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[60]
        else:
            ohlc_df = list(ohlc_df_dict.values())[-1]

        stop_price = calc_chandelier_exit_stop(ohlc_df= ohlc_df, stop_trade_type= stop_trade_type)

        return stop_price

    def calc_trail_stop_loss(self, ohlc_df_dict: dict, stop_trade_type, current_pair_price, current_order_price):

        # trailing_stop_loss_percent_threshold= 0.08
        # trailing_stop_loss_percent_change = 0.04
        #
        # new_stop_loss_price, is_update_order = calc_updated_simple_trail_stop(stop_trade_type,
        #                                                                       current_pair_price,
        #                                                                       current_order_price,
        #                                                                       trailing_stop_loss_percent_threshold,
        #                                                                       trailing_stop_loss_percent_change)

        if 60 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[60]
        else:
            ohlc_df = list(ohlc_df_dict.values())[-1]

        CE_multiplier = 3.5
        N = 30 # periods

        new_stop_loss_price = calc_chandelier_exit_stop(ohlc_df= ohlc_df, stop_trade_type= stop_trade_type)

        if new_stop_loss_price != current_order_price:
            is_update_order = True
        else:
            is_update_order = False

        return new_stop_loss_price, is_update_order

class bollinger_derivative_reversal_model_heikin_ashi:

    def __init__(self, pairs: list):

        self.pairs = pairs

        self.trigger_matrix = pd.DataFrame(data={
            'pairs': pairs,
            'bb_trade_type': [None] * len(pairs),
            'bb_trigger': [False] * len(pairs),
            'ema_deriv_trade_type': [None] * len(pairs),
            'ema_deriv_trigger': [False] * len(pairs)
        })

    def _check_bb_trigger(self, ohlc_df: pd.DataFrame, pair: str):

        '''
        Check if price had touched upper or lower bollinger band. Check the last two rows of ohlc df.

        Checking last two rows will help ensure against missing a data point that has low or a high close above/below BB.
        Particularly important for test mode because it increments forward in large chunks of time (say 30 mins).

        :param ohlc_df:
        :param pair:
        :return:
        '''

        last_row_index = ohlc_df.shape[0] - 1
        last_two_rows = ohlc_df.loc[last_row_index - 1: last_row_index + 1, :]

        trade_type = self.trigger_matrix[self.trigger_matrix.pairs == pair].bb_trade_type.to_list()[0]
        trigger = self.trigger_matrix[self.trigger_matrix.pairs == pair].bb_trigger.to_list()[0]

        is_outside_bb_range = np.where(np.logical_and(last_two_rows.HA_high > last_two_rows.HA_Bollinger_Band_High,
                                                      last_two_rows.HA_low < last_two_rows.HA_Bollinger_Band_Low),
                                       True, np.nan)
        is_touch_bb_low = np.where(last_two_rows.HA_low <= last_two_rows.HA_Bollinger_Band_Low, True, np.nan)
        is_touch_bb_high = np.where(last_two_rows.HA_high >= last_two_rows.HA_Bollinger_Band_High, True, np.nan)

        if True in is_outside_bb_range:
            trade_type = None
            trigger = False

        elif True in is_touch_bb_high:
            trade_type = 'sell'
            trigger = True

        elif True in is_touch_bb_low:
            trade_type = 'buy'
            trigger = True

        return trade_type, trigger

    def _check_ema_derivative_trigger(self, ohlc_df: pd.DataFrame, pair: str):

        # EMA_array = ohlc_df.EMA_12
        # EMA_derivative_array = utils.calc_derivative_array(EMA_array)

        EMA_derivative_array = ohlc_df.HA_ema_10_derivative.to_list()

        current_derivative = EMA_derivative_array[len(EMA_derivative_array) - 1]

        if current_derivative > 0:
            trade_type = 'buy'
            trigger = True
        elif current_derivative < 0:
            trade_type = 'sell'
            trigger = True
        else:
            trade_type = None
            trigger = False

        return trade_type, trigger

    def create_trigger_df(self, pair: str, ohlc_df_dict: dict):

        trigger_df = pd.DataFrame(data = {
            'pair': [pair],
            'trade_type': [None],
            'trigger': [False]
        })

        if 60 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[60]
        else:
            ohlc_df_list = list(ohlc_df_dict.values())
            ohlc_df = ohlc_df_list[0]

        bb_trade_type, bb_trigger = self._check_bb_trigger(pair= pair, ohlc_df= ohlc_df)
        ema_derivative_trade_type, ema_derivative_trigger = self._check_ema_derivative_trigger(pair= pair, ohlc_df= ohlc_df)

        pair_ind = self.trigger_matrix[self.trigger_matrix.pairs == pair].index[0]
        self.trigger_matrix.at[pair_ind, 'bb_trade_type'] = bb_trade_type
        self.trigger_matrix.at[pair_ind, 'bb_trigger'] = bb_trigger
        self.trigger_matrix.at[pair_ind, 'ema_deriv_trade_type'] = ema_derivative_trade_type
        self.trigger_matrix.at[pair_ind, 'ema_deriv_trigger'] = ema_derivative_trigger

        print('1.0 TRADE MODE - TRIGGER DF ---------------------------------------------' )
        print(pair)
        print(self.trigger_matrix)
        print(ohlc_df.loc[ohlc_df.shape[0]-2:ohlc_df.shape[0]-1, :])
        print('\n')

        if bb_trigger and ema_derivative_trigger == True:
            trigger = bb_trigger
            trade_type = None
            if bb_trade_type == ema_derivative_trade_type:
                trade_type = bb_trade_type
        else:
            trade_type = None
            trigger = False

        trigger_df.at[0, 'pair'] = pair
        trigger_df.at[0, 'trade_type'] = trade_type
        trigger_df.at[0, 'trigger'] = trigger

        return trigger_df

    def calc_stop_loss_unit_price(self, ohlc_df_dict: dict, stop_trade_type: str):

        if 60 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[60]
        else:
            ohlc_df = list(ohlc_df_dict.values())[0]

        stop_price = calc_chandelier_exit_stop(ohlc_df= ohlc_df, stop_trade_type= stop_trade_type)

        return stop_price

    def calc_trail_stop_loss(self, ohlc_df_dict: dict, stop_trade_type, current_pair_price, current_order_price):

        # trailing_stop_loss_percent_threshold= 0.08
        # trailing_stop_loss_percent_change = 0.04
        #
        # new_stop_loss_price, is_update_order = calc_updated_simple_trail_stop(stop_trade_type,
        #                                                                       current_pair_price,
        #                                                                       current_order_price,
        #                                                                       trailing_stop_loss_percent_threshold,
        #                                                                       trailing_stop_loss_percent_change)

        if 60 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[60]
        else:
            ohlc_df = list(ohlc_df_dict.values())[0]

        new_stop_loss_price = calc_chandelier_exit_stop(ohlc_df= ohlc_df, stop_trade_type= stop_trade_type)

        if new_stop_loss_price != current_order_price:
            is_update_order = True
        else:
            is_update_order = False

        return new_stop_loss_price, is_update_order

class trackline_follower_model():

    def __init__(self, is_heikin_ashi: bool = False):

        self.col_name_dict = {'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low',
                         'trackline': 'trackline', 'trackline_trend': 'trackline_trend',
                         'trackline_upper': 'trackline_upper', 'trackline_lower': 'trackline_lower'} \
            if is_heikin_ashi == False else \
            {'open': 'HA_open', 'close': 'HA_close', 'high': 'HA_high', 'low': 'HA_low',
             'trackline': 'HA_trackline', 'trackline_trend': 'HA_trackline_trend',
             'trackline_upper': 'HA_trackline_upper', 'trackline_lower': 'HA_trackline_lower'}

    def create_trigger_df(self, pair: str, ohlc_df_dict: dict):

        col_name_dict = self.col_name_dict

        trigger_df = pd.DataFrame(data = {
            'pair': [pair],
            'trade_type': [None],
            'trigger': [False]
        })

        if 4320 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[4320]
        else:
            ohlc_df_list = list(ohlc_df_dict.values())
            ohlc_df = ohlc_df_list[0]

        last_row = ohlc_df.shape[0] - 1
        trigger = True

        if ohlc_df.at[last_row - 1, col_name_dict['trackline_trend']] == 'up' and ohlc_df.at[last_row, col_name_dict['trackline_trend']] == 'up': # Use the 2nd last row because last row is not fully closed
            trade_type = 'buy'
        elif ohlc_df.at[last_row - 1, col_name_dict['trackline_trend']] == 'down' and ohlc_df.at[last_row, col_name_dict['trackline_trend']] == 'down':
            trade_type = 'sell'
        else:
            trade_type = None
            trigger = False

        trigger_df.at[0, 'pair'] = pair
        trigger_df.at[0, 'trade_type'] = trade_type
        trigger_df.at[0, 'trigger'] = trigger

        print('1.0 TRADE MODE - TRIGGER DF ---------------------------------------------' )
        print(pair)
        print(trigger_df)
        print('\n')

        return trigger_df

    def calc_stop_loss_unit_price(self, ohlc_df_dict: dict, stop_trade_type: str):

        if 4320 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[4320]
        else:
            ohlc_df = list(ohlc_df_dict.values())[0]

        # stop_price = calc_trackline_stop(ohlc_df=ohlc_df, stop_trade_type=stop_trade_type, trackline_col_name= self.col_name_dict['trackline'])
        stop_price = calc_chandelier_exit_stop(ohlc_df= ohlc_df, stop_trade_type= stop_trade_type)

        return stop_price

    def calc_trail_stop_loss(self, ohlc_df_dict: dict, stop_trade_type: str, current_pair_price: float, current_order_price: float):

        if 4320 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[4320]
        else:
            ohlc_df = list(ohlc_df_dict.values())[0]

        if ohlc_df.trackline_trend[ohlc_df.shape[0] - 1] == 'neutral':
            # new_stop_loss_price, is_update_order = calc_updated_trackline_trail_stop(ohlc_df= ohlc_df, current_order_price= current_order_price, col_name_input= self.col_name_dict)
            new_stop_loss_price = calc_chandelier_exit_stop(ohlc_df=ohlc_df, stop_trade_type=stop_trade_type)
            is_update_order = True

        else:

            trailing_stop_loss_percent_threshold = 0.15
            trailing_stop_loss_percent_change = 0.08

            new_stop_loss_price, is_update_order = calc_updated_simple_trail_stop(stop_trade_type= stop_trade_type,
                                                                                  current_pair_price= current_pair_price,
                                                                                  current_order_price= current_order_price,
                                                                                  trailing_stop_loss_percent_threshold= trailing_stop_loss_percent_threshold,
                                                                                  trailing_stop_loss_percent_change= trailing_stop_loss_percent_change)

        return new_stop_loss_price, is_update_order

class momentum_oscillatr_model():

    def create_trigger_df(self, pair: str, ohlc_df_dict: dict):

        col_name_dict = self.col_name_dict

        trigger_df = pd.DataFrame(data = {
            'pair': [pair],
            'trade_type': [None],
            'trigger': [False]
        })

        if 4320 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[4320]
        else:
            ohlc_df_list = list(ohlc_df_dict.values())
            ohlc_df = ohlc_df_list[0]

        last_row = ohlc_df.shape[0] - 1
        trigger = True

        if ohlc_df.at[last_row - 1, col_name_dict['trackline_trend']] == 'up' and ohlc_df.at[last_row, col_name_dict['trackline_trend']] == 'up': # Use the 2nd last row because last row is not fully closed
            trade_type = 'buy'
        elif ohlc_df.at[last_row - 1, col_name_dict['trackline_trend']] == 'down' and ohlc_df.at[last_row, col_name_dict['trackline_trend']] == 'down':
            trade_type = 'sell'
        else:
            trade_type = None
            trigger = False

        trigger_df.at[0, 'pair'] = pair
        trigger_df.at[0, 'trade_type'] = trade_type
        trigger_df.at[0, 'trigger'] = trigger

        print('1.0 TRADE MODE - TRIGGER DF ---------------------------------------------' )
        print(pair)
        print(trigger_df)
        print('\n')

        return trigger_df

    def calc_stop_loss_unit_price(self, ohlc_df_dict: dict, stop_trade_type: str):

        if 4320 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[4320]
        else:
            ohlc_df = list(ohlc_df_dict.values())[0]

        # stop_price = calc_trackline_stop(ohlc_df=ohlc_df, stop_trade_type=stop_trade_type, trackline_col_name= self.col_name_dict['trackline'])
        stop_price = calc_chandelier_exit_stop(ohlc_df= ohlc_df, stop_trade_type= stop_trade_type)

        return stop_price

    def calc_trail_stop_loss(self, ohlc_df_dict: dict, stop_trade_type: str, current_pair_price: float, current_order_price: float):

        if 4320 in ohlc_df_dict:
            ohlc_df = ohlc_df_dict[4320]
        else:
            ohlc_df = list(ohlc_df_dict.values())[0]

        if ohlc_df.trackline_trend[ohlc_df.shape[0] - 1] == 'neutral':
            # new_stop_loss_price, is_update_order = calc_updated_trackline_trail_stop(ohlc_df= ohlc_df, current_order_price= current_order_price, col_name_input= self.col_name_dict)
            new_stop_loss_price = calc_chandelier_exit_stop(ohlc_df=ohlc_df, stop_trade_type=stop_trade_type)
            is_update_order = True

        else:

            trailing_stop_loss_percent_threshold = 0.15
            trailing_stop_loss_percent_change = 0.08

            new_stop_loss_price, is_update_order = calc_updated_simple_trail_stop(stop_trade_type= stop_trade_type,
                                                                                  current_pair_price= current_pair_price,
                                                                                  current_order_price= current_order_price,
                                                                                  trailing_stop_loss_percent_threshold= trailing_stop_loss_percent_threshold,
                                                                                  trailing_stop_loss_percent_change= trailing_stop_loss_percent_change)

        return new_stop_loss_price, is_update_order

# ================================================================================================================
# STOP LOSS FUNCTIONS
# ================================================================================================================


def calc_simple_stop(ohlc_df: pd.DataFrame, stop_trade_type: str,
                     max_trade_loss_percent: float):

    '''

    :param ohlc_df:
    :param trade_type:
    :param max_trade_loss_percent: # Max allowable loss in % terms (decimal 0 to 1)
    :return:
    '''

    last_row = ohlc_df.shape[0] - 1
    current_price = ohlc_df.at[last_row, 'close']

    if stop_trade_type == 'sell':
        stop_price = current_price * (1 - max_trade_loss_percent)
    else:
        stop_price = current_price * (1 + max_trade_loss_percent)

    return stop_price


def calc_chandelier_exit_stop(ohlc_df: pd.DataFrame, stop_trade_type: str):

    last_row = ohlc_df.shape[0] - 1

    if stop_trade_type == 'buy':
        stop_price = ohlc_df.at[last_row, 'CE_upper']
    else:
        stop_price = ohlc_df.at[last_row, 'CE_lower']

    return stop_price

def calc_trackline_stop(ohlc_df: pd.DataFrame, stop_trade_type: str, trackline_col_name: str = None):

    trackline_colname = 'trackline' if trackline_col_name == None else trackline_col_name

    stop_price = ohlc_df.at[ohlc_df.shape[0] - 1, trackline_colname]

    return stop_price


def calc_updated_simple_trail_stop(stop_trade_type: str, current_pair_price: float, current_order_price: float,
                                   trailing_stop_loss_percent_threshold: float, trailing_stop_loss_percent_change: float,
                                   ohlc_df: pd.DataFrame = None):

    '''

    :param stop_trade_type:
    :param current_pair_price:
    :param current_order_price:
    :param trailing_stop_loss_percent_threshold: Percent at which trailing stop is updated in decimal i.e., 0.05 = 5%
    :param trailing_stop_loss_percent_change:  Percent at which trailing stop is updated TO in decimal i.e., 0.05 = 5% change
    :return:
    '''

    new_stop_loss_price = None
    is_update_order = False

    if stop_trade_type == 'sell' and current_pair_price > (current_order_price * (1 + trailing_stop_loss_percent_threshold)):
            new_stop_loss_price = (current_pair_price * (1 - trailing_stop_loss_percent_change))
            is_update_order = True
    elif stop_trade_type == 'buy' and current_pair_price < (current_order_price * (1 - trailing_stop_loss_percent_threshold)):
            new_stop_loss_price = (current_pair_price * (1 + trailing_stop_loss_percent_change))
            is_update_order = True

    return new_stop_loss_price, is_update_order

def calc_updated_trackline_trail_stop(ohlc_df: pd.DataFrame, current_order_price: float, col_name_input: dict):

    col_name_dict = col_name_input

    last_row = ohlc_df.shape[0] - 1

    trackline_trend_rev = list(ohlc_df[col_name_dict['trackline_trend']].to_list())
    trackline_trend_rev.reverse()
    trackline_trend_rev = np.array(trackline_trend_rev)
    non_neutral = np.where(trackline_trend_rev != 'neutral')[0]

    if non_neutral.any() == False:
        # If there is no "up" or "down" in trackline_trend. I.e., no change just neutral then don't need to update stop_loss

        new_stop_loss_price = current_order_price
        is_update_order = False

    else: # If there was an up or down trend in trackline_trend

        non_neutral_index = non_neutral[0]
        last_trend = ohlc_df.at[last_row - non_neutral_index, col_name_dict['trackline_trend']]

        is_update_order = True

        if last_trend == 'buy':
            # stop trade_type = 'sell'
            new_stop_loss_price = ohlc_df.at[last_row, col_name_dict['trackline']] * 0.9
        else:
            # stop trade_type = 'buy'
            new_stop_loss_price = ohlc_df.at[last_row, col_name_dict['trackline']] * 1.1

    if new_stop_loss_price == current_order_price:
        is_update_order = False

    return new_stop_loss_price, is_update_order

