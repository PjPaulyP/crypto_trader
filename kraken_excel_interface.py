import krakenex
import pykrakenapi as pk
import pandas as pd
import ta
import csv
import time
import utility as utils
import numpy as np
from datetime import datetime
import requests.exceptions

class KrakenExcelInterface:

    '''
    KrakenExcelInterface handles the API connection and queries
    Also read/writes the data to csv files (pricing data, account balance info, etc)
    '''

    def __init__(self, initial_time, keys_text, pairs_list, timeframe_list,
                 ohlc_csv_folder_name, trade_csv_folder_name, open_spot_positions_filename,
                 test_data_path, kraken_exchange_tester= None):

        self.initial_time = initial_time
        self.ohlc_csv_folder_name = ohlc_csv_folder_name
        self.trade_csv_folder_name = trade_csv_folder_name
        self.open_spot_positions_filename = open_spot_positions_filename
        self.test_data_path = test_data_path
        self.kraken_exchange_tester = kraken_exchange_tester
        self.pairs_list = pairs_list
        self.pairs_price_dict = {}.fromkeys(self.pairs_list, None)
        self.timeframe_list = timeframe_list
        self.api_wait_time = 6 # Wait n seconds after querying Kraken API
        self.std_kraken_timeframes = [1, 5, 15, 30, 60, 240, 1440, 10080, 21600] # These are the callable timeframes for query_ohlc function.
        self.first_timeslist_match_std_time = utils.find_first_match_of_two_lists(timeframe_list, self.std_kraken_timeframes)

        # Variables for API try/except errror handling
        self.try_exceptions = (requests.exceptions.ConnectionError)
        self.try_delay = 5
        self.try_attempts = 9

        if kraken_exchange_tester == None:
            self.is_test_mode = False
        else:
            self.is_test_mode = True

        keys_file_name = keys_text

        ### CONNECT TO KRAKEN API
        with open(keys_file_name, "r") as file:
            lines = file.read().splitlines()
            api_key = lines[0]
            api_sec = lines[1]

        api = krakenex.API()
        api.__init__(key = api_key, secret= api_sec)
        self.Kraken_API = pk.KrakenAPI(api)

        # Tade data dataframes
        self.open_orders_headers = ['refid', 'userref', 'status', 'opentm', 'starttm', 'expiretm',
                                    'vol', 'vol_exec', 'cost', 'fee', 'price', 'stopprice', 'limitprice', 'misc',
                                    'oflags', 'descr_pair', 'descr_type', 'descr_ordertype', 'descr_price',
                                    'descr_price2', 'descr_leverage', 'descr_order', 'descr_close']

        self.closed_orders_header = ['refid', 'userref', 'status', 'opentm', 'starttm', 'expiretm', 'vol', 'vol_exec',
                                     'cost', 'fee', 'price', 'stopprice', 'limitprice', 'misc', 'oflags', 'reason', 'closetm',
                                     'descr_pair', 'descr_type', 'descr_ordertype', 'descr_price', 'descr_price2',
                                     'descr_leverage', 'descr_order', 'descr_close']

        self.open_positions_header = ['ordertxid', 'posstatus', 'pair', 'time', 'type', 'ordertype', 'cost', 'fee', 'vol',
                                      'vol_closed', 'margin', 'terms', 'rollovertm', 'misc', 'oflags']
        self.account_balance_header = ['vol']


    def _query_ohlc_data(self, pair: str, timeframe: int, last_unixtime: float):

        '''
        Query and return OHLC data (Open, high, low and close pricing of a given pair and timeframe for each timestamp).

        :param pair: (str) Crypto - currency pair...ex. "SOLUSD"
        :param timeframe: (int) Timeframe of OHLC data (i.e., hourly, 4 hour or daily candles)...in units of minutes
        :param last_unixtime: (float) Unixtime of last time OHLC data was queried
        :return:
            ohlc_data (pd.DataFrame) Contains new updated OHLC data
            new_last_unixtime (float) Contains unixtime for this query timestamp
        '''

        # if timeframe not in self.std_kraken_timeframes:
        #     ohlc_data, new_last_unixtime = self._query_nonstandard_timeframe_ohlc_data(pair= pair, timeframe= timeframe, since= last_unixtime)

        if timeframe not in self.std_kraken_timeframes:
            ohlc_data, new_last_unixtime = self._query_nonstandard_timeframe_ohlc_data(pair= pair, timeframe= timeframe, since= last_unixtime)

        else:

            if self.is_test_mode == False:

                ### RETREIVE DATA FROM KRAKEN AND ADD INDEX STARTING FROM ZERO ON LEFT SIDE OF TABLE

                @utils.RetryAndCatch(exceptions_to_catch= self.try_exceptions, delay= self.try_delay, num_tries= self.try_attempts)
                def try_ohlc_data(pair, timeframe, last_unixtime):
                    ohlc_data, last = self.Kraken_API.get_ohlc_data(pair= pair, interval= timeframe, since= last_unixtime, ascending= True)
                    return ohlc_data, last

                ohlc_data, last = try_ohlc_data(pair, timeframe, last_unixtime)
                time.sleep(self.api_wait_time)
                ohlc_data = ohlc_data.reset_index()
                time_col = ohlc_data.columns.get_loc('time')
                new_last_unixtime = ohlc_data.iat[-1, time_col]

            else:

                ohlc_data, new_last_unixtime = self.kraken_exchange_tester.read_test_data(pair, timeframe, last_unixtime)

        return ohlc_data, new_last_unixtime

    def init_ohlc_data_all(self, trades_df, last_unixtime_df, timeframe):

        '''
        Calls _query_ohlc_data function and writes initial OHLC data to csv

        :param trades_df:
        :param last_unixtime_df:
        :param timeframe:
        :return:
        '''

        for i in range(trades_df.shape[0]):

            for j in range(last_unixtime_df.shape[1]):

                ohlc_init, last_unixtime_df.iat[i, j] = self._query_ohlc_data(trades_df.at[i, 'pair'], timeframe[j],
                                                                              last_unixtime_df.iat[i, j])
                ohlc_init = self._apply_indicators(ohlc_init, is_initial_data= True)
                ohlc_csv_file_name = trades_df.at[i, 'pair'] + '_' + str(timeframe[j])
                utils.write_to_csv(folder_path= self.ohlc_csv_folder_name, file_name= ohlc_csv_file_name,
                                   data_frame= ohlc_init)
                # Wait to prevent API overload
                if self.is_test_mode == False:
                    time.sleep(3)
                    
        return ohlc_init, last_unixtime_df

    def _apply_indicators(self, ohlc, is_initial_data: bool):

        '''
        Applies trade indicators to OHLC data. Data is appended on right side of existing OHLC data table.

        Further info on each indicator below...

        RSI:

            Note that the stochastic rsi input uses a non-standard RSI input. It is from TA library (Version 0.10.1, Bukosabino),
            where the stochastic RSI calc is not as per standard. It uses a different EWM based smoothing function to weight
            the last N data points vs. the standard one defined on Investopedia (https://www.investopedia.com/terms/r/rsi.asp)

            See thread https://github.com/bukosabino/ta/issues/38.

        Stochastic (RSI):

            Stochastic RSI is a momentum oscillator described by Tushar Chande and Stanley Kroll in their book The New Technical Trader.
            The aim of Stochastic RSI is to generate more Overbought and Oversold signals than Welles Wilder's original Relative Strength oscillator.

            Stochastic RSI = [RSI(14,price) - Minimum(14,RSI(14,price))] / [Maximum(14,RSI(14,price)) - Minimum(14,RSI(14,price))]

            See thread https://www.incrediblecharts.com/indicators/stochastic-rsi.php#:~:text=Stochastic%20RSI%20Formula&text=Subtract%20the%20minimum%20RSI%20value,first%20result%20by%20the%20second.

        :param ohlc: (pd.DataFrame) Pandas dataframe of OHLC (open, high, low, close) pricing data.
        :return:
            ohlc: (pd.DataFrame) Pandas dataframe of the input OHLC dataframe but with indicators applied (indicator columns
            appended on right side of OHLC data table.

        '''

        # Grab the RSI data.
        RS_up, RS_down, RSI = utils.calc_rsi(ohlc= ohlc, N= 14, is_initial_data= is_initial_data)
        BB_avg, BB_high, BB_low = utils.calc_bollinger_bands(ohlc_close= ohlc.close , N= 20, m= 2.5)

        ohlc['RS_up'] = RS_up
        ohlc['RS_down'] = RS_down
        ohlc['RSI'] = RSI
        ohlc['Stoch RSI'] = ta.momentum.stoch(close= ohlc.RSI, high= ohlc.RSI, low= ohlc.RSI, window= 14)
        ohlc['Bollinger_Band_mavg'] = BB_avg
        ohlc['Bollinger_Band_High'] = BB_high
        ohlc['Bollinger_Band_Low'] = BB_low
        ohlc['EMA_12'] = utils.calc_ema(ohlc= ohlc, N= 12, is_initial_data= is_initial_data)
        ohlc['EMA_21'] = utils.calc_ema(ohlc= ohlc, N= 21, is_initial_data= is_initial_data)
        ohlc['ema_12_derivative'] = utils.calc_derivative_array(ohlc.EMA_12, is_include_initial_val= True)

        true_range, average_true_range = utils.calc_atr(ohlc_df= ohlc, N= 3, is_initial_data= is_initial_data)
        ohlc['true_range'] = true_range
        ohlc['average_true_range'] = average_true_range
        CE_upper, CE_lower = utils.calc_chandelier_exit(ohlc_df= ohlc, CE_multiplier = 3.5, N = 3,
                                                        is_initial_data=is_initial_data)
        ohlc['CE_upper'] = CE_upper
        ohlc['CE_lower'] = CE_lower

        trackline_col_name = {'atr': 'trackline_ATR', 'high': 'high', 'low': 'low', 'close': 'close', 'open': 'open', 'CE_upper': 'trackline_upper', 'CE_lower': 'trackline_lower', \
                              'trackline': 'trackline', 'trackline_trend': 'trackline_trend'}
        trackline_TR, trackline_ATR = utils.calc_atr(ohlc_df= ohlc, N= 3, is_initial_data= is_initial_data, col_name_input= trackline_col_name)
        ohlc['trackline_TR'] = trackline_TR
        ohlc['trackline_ATR'] = trackline_ATR
        trackline_upper, trackline_lower = utils.calc_chandelier_exit(ohlc_df= ohlc, CE_multiplier = 3, N = 3,
                                                                      is_initial_data= is_initial_data, col_name_input= trackline_col_name)
        ohlc['trackline_upper'] = trackline_upper
        ohlc['trackline_lower'] = trackline_lower
        trackline, trackline_trend = utils.create_trackline(ohlc_df= ohlc, is_initial_data= is_initial_data, col_name_input= trackline_col_name)
        ohlc['trackline'] = trackline
        ohlc['trackline_trend'] = trackline_trend

        # HA_open, HA_close, HA_high, HA_low = utils.create_heiken_ashi_candles(ohlc_df= ohlc, is_initial_data= is_initial_data)
        # ohlc['HA_open'] = HA_open
        # ohlc['HA_close'] = HA_close
        # ohlc['HA_high'] = HA_high
        # ohlc['HA_low'] = HA_low
        # HA_BB_avg, HA_BB_high, HA_BB_low = utils.calc_bollinger_bands(ohlc_close= ohlc.HA_close , N= 20, m= 2.5)
        # ohlc['HA_Bollinger_Band_mavg'] = HA_BB_avg
        # ohlc['HA_Bollinger_Band_High'] = HA_BB_high
        # ohlc['HA_Bollinger_Band_Low'] = HA_BB_low
        # ohlc['HA_EMA_10'] = utils.calc_ema(ohlc= ohlc, N= 10, is_initial_data= is_initial_data, EMA_header_name= 'HA_EMA_10')
        # ohlc['HA_ema_10_derivative'] = utils.calc_derivative_array(ohlc.HA_EMA_10, is_include_initial_val= True)

        # HA_trackline_col_name = {'atr': 'HA_trackline_ATR', 'high': 'HA_high', 'low': 'HA_low', 'close': 'HA_close', 'open': 'HA_open',
        #                          'CE_upper': 'HA_trackline_upper', 'CE_lower': 'HA_trackline_lower', 'trackline': 'HA_trackline', 'trackline_trend': 'HA_trackline_trend'}
        # HA_trackline_TR, HA_trackline_ATR = utils.calc_atr(ohlc_df= ohlc, N= 3, is_initial_data= is_initial_data, col_name_input= HA_trackline_col_name)
        # ohlc['HA_trackline_TR'] = HA_trackline_TR
        # ohlc['HA_trackline_ATR'] = HA_trackline_ATR
        # HA_trackline_upper, HA_trackline_lower = utils.calc_chandelier_exit(ohlc_df= ohlc, CE_multiplier = 1.5, N = 3,
        #                                                                     is_initial_data= is_initial_data, col_name_input= HA_trackline_col_name)
        # ohlc['HA_trackline_upper'] = HA_trackline_upper
        # ohlc['HA_trackline_lower'] = HA_trackline_lower
        # HA_trackline, HA_trackline_trend = utils.create_trackline(ohlc_df= ohlc, is_initial_data= is_initial_data, col_name_input= HA_trackline_col_name)
        # ohlc['HA_trackline'] = HA_trackline
        # ohlc['HA_trackline_trend'] = HA_trackline_trend

        return ohlc

    def _append_ohlc_data(self, pair, timeframe, last_unixtime):

        # Initialize empty ohlc_data_append object
        ohlc_data_append = pd.DataFrame()

        # Query new OHLC data
        new_ohlc_data, new_last_unixtime = self._query_ohlc_data(pair, timeframe, last_unixtime)

        print('\n')
        print('3.2 TIME OHLC TS')
        print(new_last_unixtime, last_unixtime)

        # Compare if the new_last_unixtime is greater than last_unixtime
        if new_last_unixtime >= last_unixtime:

            ### SPECIFY NUMBER OF LAST ROWS TO GRAB IN EXCEL DATA
            number_of_last_data_points = 30

            ### Obtain number of rows in csv.
            ohlc_csv_file_path = self.ohlc_csv_folder_name + '/' + pair + '_' + str(timeframe) + '.csv'
            with open(ohlc_csv_file_path) as f:
                ohlc_csv_file_object = csv.reader(f)
                row_count = sum(1 for row in ohlc_csv_file_object)

            if row_count < number_of_last_data_points:
                skip_row = None
            else:
                skip_row_number = row_count - number_of_last_data_points ### Grabs last X # of data points
                skip_row = range(1, skip_row_number)

            ohlc_prev_data = pd.read_csv(filepath_or_buffer= ohlc_csv_file_path, skiprows= skip_row)
            ohlc_prev_data = ohlc_prev_data.drop(columns= ['Unnamed: 0'])
            ohlc_data_temp = pd.concat([ohlc_prev_data, new_ohlc_data], ignore_index= True).reset_index(drop= True)

            ohlc_data_temp = ohlc_data_temp.drop_duplicates(subset= 'time', keep= 'last').reset_index(drop= True)


            ### Set is_replace_last_row to True if equals last unixtime recorded on previous iteration
            if new_last_unixtime == last_unixtime:
                replace_last_N_rows = 1 # Replace last row
                append_row_number = ohlc_data_temp.shape[0] - 1 # Index of last row. Time of last row for ohlc_prev = Time of last row for new append
            else:
                replace_last_N_rows = 1 # Replace last row (update last value and add new unxtime row) c

                # Index starting where data needs to start being replaced. Grabs last row of existing ohlc and the new additional row
                append_row_number = ohlc_prev_data.shape[0] - 1


            ohlc_data_temp = self._apply_indicators(ohlc_data_temp, is_initial_data= False)
            ohlc_data_append = ohlc_data_temp[append_row_number:]

            ohlc_csv_file_name = pair + '_' + str(timeframe)
            utils.write_to_csv(folder_path= self.ohlc_csv_folder_name, file_name= ohlc_csv_file_name,
                               data_frame= ohlc_data_append, replace_last_N_rows= replace_last_N_rows)

            print('\n')
            print('3.3 TIME OHLC TS')
            print(ohlc_data_append)


        return ohlc_data_append, new_last_unixtime

    def append_ohlc_data_all(self, trades_df, last_unixtime_df, timeframe):

        for i in range(trades_df.shape[0]):

            # is_last_pair_and_timeframe = None

            for j in range(last_unixtime_df.shape[1]):

                # if i == (trades_df.shape[0] - 1) and j == (last_unixtime_df.shape[1] - 1):
                #     is_last_pair_and_timeframe = True
                ohlc_append, last_unixtime_df.iat[i, j] = self._append_ohlc_data(trades_df.at[i, 'pair'], timeframe[j],
                                                                                 last_unixtime_df.iat[i, j])
                if self.is_test_mode == False:
                    time.sleep(self.api_wait_time)
                    
        return ohlc_append, last_unixtime_df

    def get_Kraken_acc_data(self):

        '''
        Returns open orders, closed orders, open positions and account balance (cash + spot positions).
        '''


        if self.is_test_mode == False:
        ### FUNCTION NOT USED FOR TRADE_MODEL_TEST

            # Open orders
            @utils.RetryAndCatch(exceptions_to_catch=self.try_exceptions, delay=self.try_delay, num_tries=self.try_attempts)
            def try_open_orders():
                open_orders_df = self.Kraken_API.get_open_orders()
                return open_orders_df
            open_orders_df = try_open_orders()
            time.sleep(self.api_wait_time)

            # Closed orders
            @utils.RetryAndCatch(exceptions_to_catch=self.try_exceptions, delay=self.try_delay, num_tries=self.try_attempts)
            def try_closed_orders():
                closed_orders_df = self.Kraken_API.get_closed_orders(trades=True, start=self.initial_time)[0]  # Note that get_closed_orders returns a tuple (df, # of orders)
                return closed_orders_df
            closed_orders_df = try_closed_orders()
            time.sleep(self.api_wait_time)

            # Open positions (for non-spot, margin / leverage trades)
            @utils.RetryAndCatch(exceptions_to_catch=self.try_exceptions, delay=self.try_delay, num_tries=self.try_attempts)
            def try_open_positions():
                open_positions = self.Kraken_API.get_open_positions()
                return open_positions
            open_positions = try_open_positions()
            open_positions_df = pd.DataFrame(open_positions).transpose() ### Change so that unique transactions are rows and not columns
            time.sleep(self.api_wait_time)

            # Account balance (contains cash + spot positions)
            @utils.RetryAndCatch(exceptions_to_catch=self.try_exceptions, delay=self.try_delay, num_tries=self.try_attempts)
            def try_account_balance():
                account_balance_df = self.Kraken_API.get_account_balance()
                return account_balance_df
            account_balance_df = try_account_balance()
            ticker_corrected = list(map(utils.correct_if_x_ticker, account_balance_df.index.to_list()))
            account_balance_df.insert(loc=0, column='ticker_corrected', value=ticker_corrected)

            time.sleep(self.api_wait_time)

            if open_orders_df.empty:
                open_orders_df = pd.DataFrame(columns= self.open_orders_headers)
            if closed_orders_df.empty:
                closed_orders_df = pd.DataFrame(columns= self.closed_orders_header)
            if open_positions_df.empty:
                open_positions_df = pd.DataFrame(columns= self.open_positions_header)
            if account_balance_df.empty:
                account_balance_df = pd.DataFrame(columns= self.account_balance_header)

            trade_csv_file_names = ['open_orders', 'closed_orders', 'open_positions']
            all_trade_df = [open_orders_df, closed_orders_df, open_positions_df]

        else:

            open_orders_df = self.kraken_exchange_tester.open_orders_df
            closed_orders_df = self.kraken_exchange_tester.closed_orders_df
            open_positions_df = self.kraken_exchange_tester.open_positions_df

            # Account balance (contains cash + spot positions)
            account_balance_df = self.kraken_exchange_tester.account_balance_df

            trade_csv_file_names = ['open_orders', 'closed_orders', 'open_positions']
            all_trade_df = [open_orders_df, closed_orders_df, open_positions_df]

        return all_trade_df, trade_csv_file_names, account_balance_df

    def append_trade_excel_data(self, all_trade_df, trade_csv_file_names, is_first_data):

        # Writes Kraken account data to trade csv sheet.

        for i in range(len(trade_csv_file_names)):

            trade_csv_file_path = self.trade_csv_folder_name + '/' + trade_csv_file_names[i] + '.csv'

            if is_first_data == False:

                # Note that 'Closed Orders' pull from Kraken only pulls last 50 tx's. So for closed orders
                # Need to take existing closed orders, then append new closed orders
                if trade_csv_file_names[i] != 'closed_orders':

                    all_trade_df[i].to_csv(path_or_buf=trade_csv_file_path)

                else:

                    # Set a DF to as found trade CSV data. Will update this DF then write it to CSV once updated
                    temp_trade_csv_df = pd.read_csv(filepath_or_buffer=trade_csv_file_path, index_col= 0)
                    # csv_tx_id_list = temp_trade_csv_df.loc[:, 'userref'].tolist()

                    csv_tx_id_list = temp_trade_csv_df.index

                    row_where_duplicates_start = 0

                    for j in range(all_trade_df[i].shape[0]):

                        row_where_duplicates_start += 1

                        # if all_trade_df[i].at[j, 'userref'] in csv_tx_id_list:
                        if all_trade_df[i].index[j] in csv_tx_id_list:
                            # Find where duplicates start on newly queried closed position data
                            row_where_duplicates_start = j
                            break


                    temp_trade_csv_df = pd.concat([all_trade_df[i].iloc[: row_where_duplicates_start, :], temp_trade_csv_df])
                    temp_trade_csv_df.to_csv(path_or_buf= trade_csv_file_path)

            else:

                all_trade_df[i].to_csv(path_or_buf= trade_csv_file_path)

    def write_account_balance_data(self, account_balance_df: pd.DataFrame, account_balance_filename: str):

        '''
        Write account balance info to csv.

        :param account_balance_df: (pd.DataFrame) Holds the cash + spot position balance
        :param account_balance_filename: (str) Name of the account balance csv
        :return:
        '''

        account_balance_filepath = self.trade_csv_folder_name + '/' +  account_balance_filename + '.csv'
        account_balance_df.to_csv(path_or_buf= account_balance_filepath)
        self.write_open_spot_positions()

    def update_trade_account_balance(self):

        '''

        :return: (pd.DataFrame) dataframe of account's trade balances (equity, margin, unrealized pnl, etc)
        '''

        if self.is_test_mode == False:
            @utils.RetryAndCatch(exceptions_to_catch=self.try_exceptions, delay=self.try_delay, num_tries=self.try_attempts)
            def try_trade_account_balance_df():
                trade_account_balance_df = self.Kraken_API.get_trade_balance(asset='ZUSD').transpose()
                return trade_account_balance_df

            trade_account_balance_df = try_trade_account_balance_df()
            time.sleep(self.api_wait_time)

        else:
            trade_account_balance_df = self.kraken_exchange_tester.get_trade_balance()
            trade_account_balance_df.index = ['ZUSD']

        return trade_account_balance_df

    def create_order(self, order_info_df):

        order_info_df = order_info_df.reset_index(drop= True)

        if self.is_test_mode == False:

            if order_info_df.at[0, 'leverage'] == '1:1':
                leverage = 'none'
            else:
                leverage = str(order_info_df.at[0, 'leverage'][0])

            if pd.isnull(order_info_df.at[0, 'close_ordertype']):
                close_ordertype = None
            else:
                close_ordertype = str(order_info_df.at[0, 'close_ordertype'])

            if pd.isnull(order_info_df.at[0, 'close_price']):
                close_price = None
            else:
                close_price = str(order_info_df.at[0, 'close_price'])

            print('Create _order - test mode False ')
            print(order_info_df)
            print(order_info_df.at[0, 'leverage'], leverage)
            print(type(order_info_df.at[0, 'leverage']))
            print(order_info_df.at[0, 'price'])
            print(type(order_info_df.at[0, 'price']))
            print(order_info_df.at[0, 'close_price'], close_price)
            print(type(order_info_df.at[0, 'close_price']))
            print(order_info_df.at[0, 'volume'])
            print(type(order_info_df.at[0, 'volume']))
            print(order_info_df.at[0, 'close_ordertype'], close_ordertype)
            print(type(order_info_df.at[0, 'close_ordertype']))

            @utils.RetryAndCatch(exceptions_to_catch=self.try_exceptions, delay=self.try_delay, num_tries=self.try_attempts)
            def try_add_standard_order(order_info_df, leverage, close_ordertype, close_price):
                self.Kraken_API.add_standard_order(
                    ordertype= str(order_info_df.at[0, 'ordertype']),
                    type= str(order_info_df.at[0, 'type']),
                    pair= str(order_info_df.at[0, 'pair']),
                    volume= str(order_info_df.at[0, 'volume']),
                    price= str(order_info_df.at[0, 'price']),
                    leverage= leverage,
                    close_ordertype= close_ordertype,
                    close_price= close_price,
                    timeinforce= None,
                    validate= False
                )

            try_add_standard_order(order_info_df, leverage, close_ordertype, close_price)
            time.sleep(self.api_wait_time)

        else:

            self.kraken_exchange_tester.sort_new_order(order_info_df)

    def cancel_order(self, ordertxid):

        # Close limits, stop-losses that are now offsides
        if self.is_test_mode == False:

            print('Cancel open order - test mode False ')

            @utils.RetryAndCatch(exceptions_to_catch=self.try_exceptions, delay=self.try_delay, num_tries=self.try_attempts)
            def try_cancel_open_order(ordertxid):
                self.Kraken_API.cancel_open_order(txid= ordertxid)

            try_cancel_open_order(ordertxid)
            time.sleep(self.api_wait_time)

        else:

            self.kraken_exchange_tester.cancel_order(ordertxid)

    def update_open_order(self, updated_order_info_df, existing_ordertxid):

        '''
        Update existing open order, i.e., trailing stop-loss

        :param ordertxid: unique transaction ID of the open order to be cancelled
        :param order_info_df: dataframe of info for the new order (i.e., for new stop-loss)
        :return:
        '''

        if self.is_test_mode == False:

            # cancel existing order. Kraken has no update existing order functionality
            self.cancel_order(existing_ordertxid)
            # Then send new order to exchange
            self.create_order(updated_order_info_df)

        else:
            # cancel existing order. Kraken has no update existing order functionality
            self.cancel_order(existing_ordertxid)

            # Then send new order to exchange
            self.kraken_exchange_tester.sort_new_order(updated_order_info_df)

    def loop_kraken_interface_functions(self, new_order_info_df, close_order_info_df, cancel_orders_df,
                                        updated_stop_orders_df, cancel_traiL_stop_orders_df):

        for row in range(cancel_orders_df.shape[0]):
            self.cancel_order(cancel_orders_df.at[row, 'ordertxid'])
        for row in range(close_order_info_df.shape[0]):
            self.create_order(close_order_info_df.iloc[row].to_frame().transpose())
        for row in range(new_order_info_df.shape[0]):
            self.create_order(new_order_info_df.iloc[row].to_frame().transpose())
        for row in range(cancel_traiL_stop_orders_df.shape[0]):
            self.cancel_order(cancel_traiL_stop_orders_df.at[row, 'ordertxid'])
        for row in range(updated_stop_orders_df.shape[0]):
            self.create_order(updated_stop_orders_df.iloc[row].to_frame().transpose())

        if self.is_test_mode == True:
            print('1.0 Kraken Excel Interface Loop Functions')
            print(self.kraken_exchange_tester.open_orders_df)
            print(self.kraken_exchange_tester.open_positions_df)
            print(self.kraken_exchange_tester.account_balance_df)

    def get_pair_price(self, timeframe: int = 240):

        if self.is_test_mode == False:

            for row in range(len(self.pairs_list)):
                csv_file_path = self.ohlc_csv_folder_name + '/' + self.pairs_list[row] + '_' + str(timeframe) + '.csv'
                csv_data = pd.read_csv(csv_file_path)
                latest_price = float(csv_data.at[csv_data.shape[0] - 1, 'close'])
                self.pairs_price_dict[
                    list(self.pairs_price_dict.keys())[row]] = latest_price

        else:
            self.pairs_price_dict = self.kraken_exchange_tester.get_pair_price()

        return self.pairs_price_dict

    def write_open_spot_positions(self):

        open_spot_positions_df = utils.read_csv_data(self.trade_csv_folder_name, 'account_balance')
        open_spot_positions_df.rename(columns={'Unnamed: 0': 'ticker'}, inplace=True)
        open_spot_positions_df = open_spot_positions_df[open_spot_positions_df.ticker != 'ZUSD']
        open_spot_positions_df = open_spot_positions_df[open_spot_positions_df.vol != 0]
        open_spot_positions_df = open_spot_positions_df.reset_index(drop= True)
        open_spot_positions_df['entry_price'] = None

        # Read closed position data. Note that df starts with most recent closed positions at top of table
        closed_positions_df = utils.read_csv_data(self.trade_csv_folder_name, 'closed_orders')

        for row in range(open_spot_positions_df.shape[0]):

            ticker = open_spot_positions_df.at[row, 'ticker_corrected']
            volume = open_spot_positions_df.at[row, 'vol']
            cumulative_vol = 0

            relevant_closed_positions_df = pd.DataFrame()

            for sub_row in range(closed_positions_df.shape[0]):

                sub_pair = closed_positions_df.at[sub_row, 'descr_pair']
                sub_ticker = utils.convert_pair_to_ticker(sub_pair)
                sub_volume = closed_positions_df.at[sub_row, 'vol_exec']
                sub_order_status = closed_positions_df.at[sub_row, 'status']
                sub_type = closed_positions_df.at[sub_row, 'descr_type']
                sub_leverage = closed_positions_df.at[sub_row, 'descr_leverage']
                if sub_leverage == '1:1':
                    sub_leverage = 'none' # make it match non-test

                if sub_order_status == 'closed' and \
                        sub_ticker == ticker and \
                        sub_type == 'buy' and \
                        sub_leverage == 'none':

                    cumulative_vol += sub_volume

                    this_sub_row = closed_positions_df.iloc[sub_row].to_frame().transpose()
                    relevant_closed_positions_df = pd.concat([relevant_closed_positions_df, this_sub_row])

                    if cumulative_vol == volume:

                        avg_entry_price = ((relevant_closed_positions_df.vol_exec * relevant_closed_positions_df.price).sum()) / cumulative_vol
                        open_spot_positions_df.at[row, 'entry_price'] = avg_entry_price

                        break

        file_path = self.trade_csv_folder_name + '/' + self.open_spot_positions_filename + '.csv'
        open_spot_positions_df.to_csv(file_path)

    def _query_nonstandard_timeframe_ohlc_data(self, pair, timeframe, since):

        ohlc_df = pd.DataFrame(columns=['dtime', 'time', 'open', 'high', 'low', 'close', 'volume'])

        shortest_timeframe = min(self.timeframe_list)
        shortest_timeframe_file_name = pair + '_' + str(shortest_timeframe)

        ohlc_shortest_timeframe_df = utils.read_csv_data(csv_folder_name= self.ohlc_csv_folder_name,
                                                         csv_file_name= shortest_timeframe_file_name)
        ohlc_shortest_timeframe_df = ohlc_shortest_timeframe_df.reset_index(drop= True)

        first_time = ohlc_shortest_timeframe_df.at[0, 'time']
        last_time = ohlc_shortest_timeframe_df.at[ohlc_shortest_timeframe_df.shape[0] -1, 'time']
        previous_timeframe_interval = first_time - (first_time % (timeframe * 60))
        starting_timeframe_interval = previous_timeframe_interval + (timeframe * 60)
        last_timeframe_interval = last_time - (last_time % (timeframe * 60))
        timeframe_sec = timeframe * 60

        time_list = np.arange(starting_timeframe_interval, last_timeframe_interval, timeframe_sec).tolist()

        for row in range(len(time_list)):

            loop_current_time = time_list[row]
            loop_next_time = loop_current_time + (timeframe * 60)

            # Determine row from ohlc data closest to current time
            row_closest_to_current_loop_time = ohlc_shortest_timeframe_df.iloc[
                (ohlc_shortest_timeframe_df.time - loop_current_time).abs().argsort()[:1]]
            row_closest_to_current_loop_time = row_closest_to_current_loop_time.reset_index(drop=True)

            # Determine row from ohlc data closest to next time
            row_closest_to_next_loop_time = ohlc_shortest_timeframe_df.iloc[
                (ohlc_shortest_timeframe_df.time - loop_next_time).abs().argsort()[:1]]
            row_closest_to_next_loop_time = row_closest_to_next_loop_time.reset_index(drop=True)

            open = row_closest_to_current_loop_time.at[0, 'close']
            close = row_closest_to_next_loop_time.at[0, 'close']

            high = ohlc_shortest_timeframe_df.loc[
                ohlc_shortest_timeframe_df.time.between(loop_current_time, loop_next_time)].high.max()

            low = ohlc_shortest_timeframe_df.loc[
                ohlc_shortest_timeframe_df.time.between(loop_current_time, loop_next_time)].low.min()

            ohlc_df.at[row, 'open'] = open
            ohlc_df.at[row, 'high'] = high
            ohlc_df.at[row, 'low'] = low
            ohlc_df.at[row, 'close'] = close

            ohlc_df.at[row, 'dtime'] = datetime.utcfromtimestamp(loop_current_time).strftime('%Y-%m-%d %H:%M:%S')
            ohlc_df.at[row, 'time'] = loop_current_time

        new_last_unixtime = ohlc_df.at[ohlc_df.shape[0] - 1, 'time']

        if since == None:
            since_unixtime = ohlc_df.at[0, 'time']
        else:
            since_unixtime = since

        ohlc_df = ohlc_df.loc[ohlc_df.time.between(since_unixtime, new_last_unixtime)]

        return ohlc_df, new_last_unixtime

    def update_trade_asset_info(self, pairs):

        # Intialize min volume size for each trading pair
        # Source https://support.kraken.com/hc/en-us/articles/205893708-Minimum-order-size-volume-for-trading

        # Initialize price decimal places
        # Source https://support.kraken.com/hc/en-us/articles/4521313131540-Price-and-volume-decimal-precision#:~:text=E.g.%20Maximum%20price%20precision%20for,BTC%2C%20but%20not%200.002000001%20BTC.

        pair_min_trade_volume_dict ={}
        pair_max_trade_price_decimal ={}

        for row in range(len(pairs)):

            @utils.RetryAndCatch(exceptions_to_catch=self.try_exceptions, delay=self.try_delay, num_tries=self.try_attempts)
            def try_tradable_asset_pairs(pair):
                pair_info = self.Kraken_API.get_tradable_asset_pairs(pair= pair)
                return pair_info

            pair_info = try_tradable_asset_pairs(pair=pairs[row])

            # pair_min_trade_volume_dict[pairs[row]] = float(pair_info.at[pairs[row], 'ordermin'])
            # pair_max_trade_price_decimal[pairs[row]] = int(pair_info.at[pairs[row], 'pair_decimals'])

            # Use the "altname" column to identify correct row - the pair symbols used in the index are not standard on Kraken
            pair_min_trade_volume_dict[pairs[row]] = float(pair_info.ordermin[pair_info.altname == pairs[row]].reset_index(drop= True)[0])
            pair_max_trade_price_decimal[pairs[row]] = int(pair_info.pair_decimals[pair_info.altname == pairs[row]].reset_index(drop=True)[0])

            time.sleep(self.api_wait_time)

        return pair_min_trade_volume_dict, pair_max_trade_price_decimal





