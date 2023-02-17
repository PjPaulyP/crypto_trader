import pandas as pd
from datetime import datetime
import utility as utils
import re
import csv
import sys

class kraken_exchanger_tester:

    ### CLASS INTENDED TO SERVE AS MOCK KRAKEN EXCHANGE TO
    ### BACKTEST PRIVATE FUNCTIONALITY AND TRADE MODEL PERFORMANCE

    def __init__(self, ohlc_csv_test_data_folder_path, trade_csv_folder_path, pairs_list, initial_cash_balance, timeframes_list: list):

        ### CREATE MOCK KRAKEN DATA INFO PULLED FROM EXCHANGE

        open_orders_headers = ['refid', 'userref', 'status', 'opentm', 'starttm', 'expiretm',
                               'vol', 'vol_exec', 'cost', 'fee', 'price', 'stopprice', 'limitprice', 'misc',
                               'oflags', 'descr_pair', 'descr_type', 'descr_ordertype', 'descr_price',
                               'descr_price2', 'descr_leverage', 'descr_order', 'descr_close']

        closed_orders_header = ['refid', 'userref', 'status', 'opentm', 'starttm', 'expiretm', 'vol', 'vol_exec',
                                'cost', 'fee', 'price', 'stopprice', 'limitprice', 'misc', 'oflags', 'reason', 'closetm',
                                'descr_pair', 'descr_type', 'descr_ordertype', 'descr_price', 'descr_price2',
                                'descr_leverage', 'descr_order', 'descr_close']

        open_positions_header = ['ordertxid', 'posstatus', 'pair', 'time', 'type', 'ordertype', 'cost', 'fee', 'vol',
                                 'vol_closed', 'margin', 'terms', 'rollovertm', 'misc', 'oflags']

        self.first_test_time = 0
        # Set last_test_time to now, so that we can overwrite it in _initialize_test_times function later with earliest last test time
        self.last_test_time = datetime.timestamp(datetime.now())
        self.current_test_time_first_iteration_percent = 0.10
        self.first_iteration_end_time = 0
        self.current_test_time = 0
        # self.test_time_increment = min(timeframes_list) * 60 / 2 # (seconds)...increment by half of min timeframe duration (i.e., 1 Hr = increment 30 mins...)
        self.test_time_increment = 1800
        self.pairs_list = pairs_list
        self.pairs_price_dict = {}.fromkeys(self.pairs_list, None)


        # Dict of complete OHLC history for each pair
        self.all_test_ohlc_df_dict = {}

        self.open_orders_df = pd.DataFrame(columns= open_orders_headers)
        self.closed_orders_df = pd.DataFrame(columns= closed_orders_header)
        self.open_positions_df = pd.DataFrame(columns= open_positions_header)

        self.trade_csv_folder_path = trade_csv_folder_path
        self.ohlc_csv_test_data_folder_path = ohlc_csv_test_data_folder_path

        self.trade_csv_file_names = ['open_orders', 'closed_orders', 'open_positions']
        self.all_trade_df = [self.open_orders_df, self.closed_orders_df, self.open_positions_df]

        # Trade Account Balance Variables
        self.trade_balance = 0
        self.margin_amount = 0
        self.unrealized_PnL = 0
        self.open_position_cost_basis = 0
        self.open_position_current_floating_valuation = 0
        self.equity = 0
        self.free_margin = 0
        self.margin_level = 0

        self.fee = 0.003 # Assume fee is 0.3%

        # Initialize Account balance (contains cash + spot positions)
        pairs_list_acc_bal = list(pairs_list)
        for pair in range(len(pairs_list_acc_bal)):
            pairs_list_acc_bal[pair] = pairs_list_acc_bal[pair].replace('USD', '')
        pairs_list_acc_bal.append('ZUSD')
        self.account_balance_df = pd.DataFrame(index= pairs_list_acc_bal, columns= ['vol']).fillna(0)
        self.account_balance_df.at['ZUSD', 'vol'] = initial_cash_balance
        ticker_corrected = list(map(utils.correct_if_x_ticker, self.account_balance_df.index.to_list()))
        self.account_balance_df.insert(loc=0, column='ticker_corrected', value=ticker_corrected)

        self._initialize_test_times()

        # Push all historical OHLC data for each pair into class variable
        self.read_test_data_to_dict()

    def _initialize_test_times(self):

        for i in range(len(self.pairs_list)):

            csv_file_path = self.ohlc_csv_test_data_folder_path + '/' + self.pairs_list[i] + '.csv'
            with open(csv_file_path, 'r') as file:
                csv_read = csv.reader(file)
                csv_read = list(csv_read)
                first_unixtime = csv_read[0][0]
                last_unixtime = csv_read[-1][0]

            if float(first_unixtime) > self.first_test_time:
                self.first_test_time= float(first_unixtime)
            if float(last_unixtime) < self.last_test_time:
                self.last_test_time = float(last_unixtime)

        self.current_test_time = self.first_test_time
        self.first_iteration_end_time = self.first_test_time + (self.current_test_time_first_iteration_percent * (self.last_test_time - self.first_test_time))

    def increment_test_time(self, updated_current_unixtime: float= None):

        if updated_current_unixtime == None:
            self.current_test_time += self.test_time_increment
        else:
            self.current_test_time = updated_current_unixtime

        if self.current_test_time >= self.last_test_time:
            sys.exit()

    def open_position(self, order_info_df, unique_tx_id):

        ### OPEN POSITION

        for row in range(order_info_df.shape[0]):

            leverage = order_info_df.at[row, 'leverage']
            pair = order_info_df.at[row, 'pair']
            cost = order_info_df.at[row, 'price'] * order_info_df.at[row, 'volume']
            volume = float(order_info_df.at[row, 'volume'])
            fee = cost * self.fee

            print('2.1 ORDER PRICE T/S')
            print(pair, cost, volume)

            if leverage == '1:1':

                # Opening position will reduce cash position by the posted margin amount
                ticker = pair.replace('USD', '')
                self.account_balance_df.at[ticker, 'vol'] += volume

                trade_val = cost


            elif leverage != '1:1':

                ### MOCK SEND ORDER FUNC AND UPDATE OPEN POSITION DATAFRAME IN CLASS OBJECT

                order_info_df_final = pd.DataFrame({
                    'ordertxid': [unique_tx_id],
                    'posstatus': ['open'],
                    'pair': [pair],
                    'time': [self.current_test_time],
                    'type': [order_info_df.at[row, 'type']],
                    'ordertype': [order_info_df.at[row, 'ordertype']],
                    'cost': [cost], ### cost = price * vol here
                    'fee': [fee], ### Market order ~0.002, Limit Order ~0.001
                    'vol': [order_info_df.at[row, 'volume']],
                    'vol_closed': [0],
                    'margin': [
                        order_info_df.at[row, 'price'] * order_info_df.at[row, 'volume'] / \
                        float(order_info_df.at[row, 'leverage'][0])],
                    'terms': [None],
                    'rollovertm': [None],
                    'misc': [None],
                    'oflags': [None]})

                self.open_positions_df = pd.concat([self.open_positions_df, order_info_df_final], ignore_index= True)

                trade_val = order_info_df_final.at[0, 'margin']

            # Opening position will reduce cash position by the posted margin amount
            self.update_cash_and_free_margin_after_trade(is_new_position=True, trade_val= trade_val, fee= fee)

    def process_executed_order(self, order_info_df, unique_tx_id= None):

        # First, close existing open positions based on volume.
        close_vol_remaining = order_info_df.at[0, 'volume']

        # If leverage is 1:1, close spot position (account balance), otherwise close leveraged open position
        leverage = order_info_df.at[0, 'leverage']

        if leverage == '1:1' and order_info_df.at[0, 'type'] == 'sell':
            close_vol_remaining = self.close_open_spot_positions(order_info_df= order_info_df)
        elif leverage != '1:1':
            close_vol_remaining = self.close_open_trade_positions(order_info_df= order_info_df)

        # Check how much volume actually got executed (i.e., was an open position closed? partially closed?)
        vol_exec = order_info_df.at[0, 'volume'] - close_vol_remaining

        # If some or all position was closed, then add to close order the portion of the order that got executed
        if vol_exec != 0:
            order_info_df.at[0, 'volume'] = vol_exec
            close_order_add_df = self.create_closed_order_row(order_info_df, unique_tx_id)
            self.closed_orders_df = pd.concat([close_order_add_df, self.closed_orders_df])  # Append new close order row. Recent orders displayed first

        # Open up new position for remaining vol, if any. For fresh new positions (i.e., non-close), order would go straight here
        if close_vol_remaining > 0:
            remainder_order_info_df = order_info_df
            remainder_order_info_df.at[0, 'volume'] = close_vol_remaining
            remainder_order_unique_tx_id = utils.create_order_tx_id(remainder_order_info_df.at[0, 'pair'], remainder_order_info_df.at[0, 'type'])
            self.open_position(remainder_order_info_df, remainder_order_unique_tx_id)
            close_order_add_df = self.create_closed_order_row(remainder_order_info_df, remainder_order_unique_tx_id)
            self.closed_orders_df = pd.concat([close_order_add_df, self.closed_orders_df])  # Append new close order row. Recent orders displayed first

        ### Create stop loss open order if order calls for it
        if order_info_df.at[0, 'close_ordertype'] == 'stop-loss':

            ### Make the associated stop loss the opposite direction of initial order
            if order_info_df.at[0, 'type'] == 'buy':
                close_order_type = 'sell'
            else:
                close_order_type = 'buy'

            ### SET STOP LOSS ORDER = TO INITIAL ORDER AND OVERWRITE APPLICABLE PARAMTERS
            stop_order_info_df = order_info_df
            stop_order_info_df.at[0, 'ordertype'] = 'stop-loss'
            stop_order_info_df.at[0, 'type'] = close_order_type
            stop_order_info_df.at[0, 'price'] = order_info_df.at[0, 'close_price']
            stop_order_info_df.at[0, 'close_ordertype'] = None

            self.open_order(stop_order_info_df)

    def close_open_trade_positions(self, order_info_df):

        # Set variable to size of open positions df
        open_positions_df_num_rows = self.open_positions_df.shape[0]

        # Create copy of self.open_positions for editing reasons. Then set self.open_positions equal
        open_positions_df_temp = self.open_positions_df.reset_index(drop=True)

        close_vol_remaining = order_info_df.at[0, 'volume']
        pair_order = order_info_df.at[0, 'pair']

        for row_open_position in range(open_positions_df_num_rows):

            pair_open_position = open_positions_df_temp.at[row_open_position, 'pair']
            type_open_position = open_positions_df_temp.at[row_open_position, 'type']
            price_open_position = open_positions_df_temp.at[row_open_position, 'cost'] / open_positions_df_temp.at[row_open_position, 'vol']
            price_current = order_info_df.at[0, 'price']

            ### IF OPEN POSITION == SAME PAIR && OPEN POSITION == OPPOSITE OF CLOSE TRADE TYPE
            if pair_open_position == pair_order and type_open_position != order_info_df.at[0, 'type']:

                vol_open = open_positions_df_temp.at[row_open_position, 'vol'] - open_positions_df_temp.at[row_open_position, 'vol_closed']
                close_vol_remaining_temp = close_vol_remaining - vol_open

                if close_vol_remaining_temp >= 0:

                    # Set close vol remaining (subtracted the vol_open of current existing position row). Then set vol_closed = vol for row
                    close_vol_remaining = close_vol_remaining_temp
                    open_positions_df_temp.at[row_open_position, 'vol_closed'] = open_positions_df_temp.at[row_open_position,'vol']
                    trade_margin = open_positions_df_temp.at[row_open_position, 'margin']

                    # Drop the row from existing open position
                    open_positions_df_temp= open_positions_df_temp.drop(row_open_position) # IF CLOSE VOL REMAINING > 0, DROP ROW FROM OPEN POSITIONS DF

                    vol_executed = vol_open

                else:

                    open_positions_df_temp.at[row_open_position, 'vol_closed'] = open_positions_df_temp.at[row_open_position, 'vol_closed'] + close_vol_remaining
                    vol_executed = close_vol_remaining
                    close_vol_remaining = 0
                    trade_margin = open_positions_df_temp.at[row_open_position, 'margin'] * (vol_executed / open_positions_df_temp.at[row_open_position, 'vol'])
                # Calculate how much cash the position will give back if closing it (margin + pnl)

                position_pnl = utils.calculate_position_pnl(trade_type= type_open_position,
                                                            trade_opening_price= price_open_position,
                                                            pair_current_price= price_current,
                                                            trade_current_vol= vol_executed)
                close_trade_val = trade_margin + position_pnl
                fee = (order_info_df.at[0, 'price'] * vol_open) * self.fee # Fee applies to total_open_vol * price

                self.update_cash_and_free_margin_after_trade(is_new_position= False, trade_val= close_trade_val, fee= fee)

        open_positions_df_temp = open_positions_df_temp.reset_index(drop= True)
        self.open_positions_df = open_positions_df_temp

        return close_vol_remaining

    def close_open_spot_positions(self, order_info_df):

        pair = order_info_df.at[0, 'pair']

        # Opening position will reduce cash position by the posted margin amount
        ticker = pair.replace('USD', '')

        order_vol = order_info_df.at[0, 'volume']
        initial_account_vol = self.account_balance_df.at[ticker, 'vol']
        self.account_balance_df.at[ticker, 'vol'] -= order_vol
        close_vol_remaining = order_vol - initial_account_vol

        close_trade_val = order_info_df.at[0, 'price'] * order_vol
        fee = (order_info_df.at[0, 'price'] * order_vol) * self.fee  # Fee applies to total_open_vol * price

        self.update_cash_and_free_margin_after_trade(is_new_position= False, trade_val= close_trade_val, fee= fee)

        return close_vol_remaining

    def create_closed_order_row(self, order_info_df, unique_tx_id):

        if order_info_df.at[0, 'ordertype'] == 'stop-loss':
            descr_price = order_info_df.at[0, 'price']
            stopprice = order_info_df.at[0, 'close_price']
        else:
            descr_price = 0
            stopprice = 0

        if order_info_df.at[0, 'ordertype'] == ('limit'):
            descr_price = order_info_df.at[0, 'price']
        else:
            descr_price = 0


        cost = order_info_df.at[0, 'price'] * order_info_df.at[0, 'volume']


        close_order_df = pd.DataFrame(
            index= [unique_tx_id],
            data= {
                'userref': [None],
                'status': ['closed'],
                'opentm': [self.current_test_time],
                'starttm': [0],
                'expiretm': [0],
                'vol': [order_info_df.at[0, 'volume']],
                'vol_exec': [order_info_df.at[0, 'volume']], # Assuming market order, or stop-loss
                'cost': [cost],  ### cost = price * vol here
                'fee': [cost * self.fee],
                'price': [order_info_df.at[0, 'price']],
                'stopprice': [stopprice],  ### REVISIT THIS LATER...CODE IN STOP LOSS INFO
                'limitprice': [0],
                'misc': [None],
                'oflags': [None],
                'reason': [None],
                'closetm': [self.current_test_time],
                'descr_pair': [order_info_df.at[0, 'pair']],
                'descr_type': [order_info_df.at[0, 'type']],
                'descr_ordertype': [order_info_df.at[0, 'ordertype']],
                'descr_price': [descr_price],
                'descr_price2': [0],
                'descr_leverage': [order_info_df.at[0, 'leverage']],
                'descr_order': [None],
                'descr_close': [None]
            }
        )

        return close_order_df

    def cancel_order(self, ordertxid):

        open_orders_df_temp = self.open_orders_df.reset_index()
        open_orders_df_temp.rename(columns={'index': 'index_col'}, inplace=True)
        open_orders_df_temp_change = open_orders_df_temp

        if ordertxid in open_orders_df_temp.index_col.to_list():

            # Find row of unique tx id in open orders. Keep as a list for the statement below
            row_of_unique_tx_id = open_orders_df_temp.index_col[open_orders_df_temp.index_col == ordertxid].index.to_list()

            # Keep row_of_unique_tx_id as a list. If something is in list, then drop that row on the "change" version
            # of open orders df temp.
            if row_of_unique_tx_id:
                open_orders_df_temp_change = open_orders_df_temp_change.drop(row_of_unique_tx_id[0])

            # Edit open_orders_df_temp_change and set self.open_orders_df to equal
            open_orders_df_temp_change.index = open_orders_df_temp_change.index_col.to_list()
            open_orders_df_temp_change = open_orders_df_temp_change.drop(columns=['index_col'])
            self.open_orders_df = open_orders_df_temp_change

        return self.open_orders_df

    def open_order(self, order_info_df, unique_tx_id= None):

        if unique_tx_id == None:
            unique_tx_id = utils.create_order_tx_id(order_info_df.at[0, 'pair'], order_info_df.at[0, 'type'])

        ### CREATE OPEN ORDER (ADD OPEN ORDER TO OPEN ORDERS DF)
        ### USE TO CREATE STOP LOSS OPEN ORDERS

        for row in range(order_info_df.shape[0]):

            open_order_df = pd.DataFrame(
                index= [unique_tx_id],
                data= {
                    'userref': [None],
                    'status': ['open'],
                    'opentm': [self.current_test_time],
                    'starttm': [0],
                    'expiretm': [0],
                    'vol': [order_info_df.at[row, 'volume']],
                    'vol_exec': [0],
                    'cost': [0], ### IS THIS VOL_EXEC * PRICE?
                    'fee': [0], ### BELIEVE THIS IS ACTUAL EXECUTION VALUE
                    'price': [0], ### BELIEVE THIS IS ACTUAL EXECUTION VALUE
                    'stopprice': [0], ### BELIEVE THIS IS ACTUAL EXECUTION VALUE
                    'limitprice': [0], ### BELIEVE THIS IS ACTUAL EXECUTION VALUE
                    'misc': [None],
                    'oflags': [None],
                    'descr_pair': [order_info_df.at[row, 'pair']],
                    'descr_type': [order_info_df.at[row, 'type']],
                    'descr_ordertype': [order_info_df.at[row, 'ordertype']],
                    'descr_price': [order_info_df.at[row, 'price']],
                    'descr_price2': [0],
                    'descr_leverage': [order_info_df.at[row, 'leverage']],
                    'descr_order': [None],
                    'descr_close': [None]
                }
            )

            # Append new open order. Newest open orders display first
            self.open_orders_df = pd.concat([open_order_df, self.open_orders_df])

    def execute_open_order(self, pair, current_price):

        ### CHECK IF CONDITIONS ARE MET TO TRIGGER AN OPEN ORDER

        open_orders_df_temp = self.open_orders_df.reset_index()
        open_orders_df_temp.rename(columns={'index': 'index_col'}, inplace=True)

        # Create a copy of open_orders_df_temp - this will be used to drop
        # open orders as needed after being executed
        open_orders_df_temp_change = open_orders_df_temp

        ### LOOP THROUG THE EXISTING OPEN ORDERS TO SEE IF TRIGGER LOGIC MET
        for row in range(open_orders_df_temp.shape[0]):

            open_trigger = False

            if open_orders_df_temp.at[row, 'descr_pair'] == pair:

                ### LIMIT FOR SHORT
                if open_orders_df_temp.at[row, 'descr_type'] == 'sell' \
                        and open_orders_df_temp.at[row, 'descr_ordertype'] == 'limit' \
                        and open_orders_df_temp.at[row, 'descr_price'] <= current_price:
                    open_trigger = True

                ### STOP LOSS FOR LONG
                elif open_orders_df_temp.at[row, 'descr_type'] == 'sell' \
                        and open_orders_df_temp.at[row, 'descr_ordertype'] == 'stop-loss'\
                        and open_orders_df_temp.at[row, 'descr_price'] >= current_price:
                    open_trigger = True

                ### LIMIT FOR LONG
                elif open_orders_df_temp.at[row, 'descr_type'] == 'buy' \
                        and open_orders_df_temp.at[row, 'descr_ordertype'] == 'limit'\
                        and open_orders_df_temp.at[row, 'descr_price'] >= current_price:
                    open_trigger = True

                ### STOP LOSS FOR LONG
                elif open_orders_df_temp.at[row, 'descr_type'] == 'buy' \
                        and open_orders_df_temp.at[row, 'descr_ordertype'] == 'stop-loss' \
                        and open_orders_df_temp.at[row, 'descr_price'] <= current_price:
                    open_trigger = True

                if open_trigger == True:

                    print('1.0 OPEN ORDER TRIGGERED -----------------')
                    print('1.1 PAIR -----------------')
                    print(pair)
                    print('1.2 CURRENT PRICE -----------------')
                    print(current_price)
                    print('1.3 OPEN ORDER ROW -----------------')
                    print(open_orders_df_temp.iloc[row])

                    ### READ THE STOP LOSS INFO FROM THE 'descr_close' COLUMN OF EXCEL SHEET.
                    # Below is for stop-loss orders.
                    if open_orders_df_temp.at[row, 'descr_close'] == None:
                        close_ordertype_temp = None
                        stop_price_temp = None
                    # Below is for limit orders w/ stop loss description in descr_close
                    elif 'stop loss' in open_orders_df_temp.at[row, 'descr_close']:
                        close_ordertype_temp = 'stop loss'
                        close_order_descr = open_orders_df_temp.at[row, 'descr_close']
                        r = re.compile(r"\S\d*[.]?\d\S*")
                        close_order_descr_numList = r.findall(close_order_descr)
                        stop_price_temp = close_order_descr_numList[-1] ### ASSUMING STOP PRICE IS ALWAYS THE LAST NUMBER AS PER SAMPLE

                    remaining_order_volume = open_orders_df_temp.at[row, 'vol'] - open_orders_df_temp.at[row, 'vol_exec']

                    if open_orders_df_temp.at[row, 'descr_ordertype'] == 'stop-loss':
                        # order_price = current_price # Commented this out, will update order_price to current price in sort_new_order
                        order_price = open_orders_df_temp.at[row, 'descr_price'] # Keep this as order price - sort_new_order will update order price to curr price

                    else:
                        order_price = open_orders_df_temp.at[row, 'descr_price']

                    order_info_df = pd.DataFrame({
                        'ordertype': [open_orders_df_temp.at[row, 'descr_ordertype']],
                        'type': [open_orders_df_temp.at[row, 'descr_type']],
                        'pair': [open_orders_df_temp.at[row, 'descr_pair']],
                        'volume': [remaining_order_volume],
                        'price': [order_price],
                        'leverage': [open_orders_df_temp.at[row, 'descr_leverage']],
                        'close_ordertype': [close_ordertype_temp],
                        'close_price': [stop_price_temp]
                    })

                    #### CHECK OPEN ORDERS AGAIN TO SEE HOW MUCH VOLUME ACTUALLY GOT FILLED
                    ### THEN UPDATE OPEN ORDERS DF ACCORDINGLY
                    actual_vol_exec = open_orders_df_temp.at[row, 'vol_exec']
                    unique_tx_id = open_orders_df_temp.at[row, 'index_col']

                    # Find row of unique tx id in open orders. Keep as a list for the statement below
                    row_of_unique_tx_id = open_orders_df_temp.index_col[open_orders_df_temp.index_col == unique_tx_id].index.to_list()

                    # Only execute order and/or drop row from open_orders if the "Cancel Order" didn't get to it already.
                    # Keep row_of_unique_tx_id as a list
                    if row_of_unique_tx_id:

                        if actual_vol_exec == 0:

                            print('1.0 EXEC OPEN ORDER TS ------------------------')
                            print(row_of_unique_tx_id)
                            print(row_of_unique_tx_id[0])
                            print(type(row_of_unique_tx_id))

                            open_orders_df_temp_change = open_orders_df_temp_change.drop(row_of_unique_tx_id[0]) ### IF VOL REMAINING = 0, DROP ROW FROM OPEN POSITIONS DF
                        self.sort_new_order(order_info_df= order_info_df, unique_tx_id= unique_tx_id)

        open_orders_df_temp_change.index = open_orders_df_temp_change.index_col.to_list()
        open_orders_df_temp_change = open_orders_df_temp_change.drop(columns= ['index_col'])
        self.open_orders_df = open_orders_df_temp_change

    def sort_new_order(self, order_info_df, unique_tx_id= None):

        ### Sort new order - i.e., should it go to open order or execute?

        order_info_df = order_info_df.reset_index(drop= True)
        open_order_df_temp = self.open_orders_df.reset_index()
        open_order_df_temp.rename(columns={'index': 'index_col'}, inplace=True)

        for row in range(order_info_df.shape[0]):

            if unique_tx_id == None:
                unique_tx_id = utils.create_order_tx_id(order_info_df.at[row, 'pair'], order_info_df.at[row, 'type'])

            if order_info_df.at[row, 'ordertype'] == 'market' or \
                    (order_info_df.at[row, 'ordertype'] == 'stop-loss'and
                     utils.determine_stop_loss_execute(order_info_df, self.pairs_price_dict[order_info_df.at[row, 'pair']]) == True) :
                print('PROCESS EXECUTED ORDER!!!')

                # If order is a stop-loss order being executed, then update executed price to current market price
                if order_info_df.at[row, 'ordertype'] == ('stop-loss' or 'market'):
                    order_info_df.at[row, 'price'] = self.pairs_price_dict[order_info_df.at[row, 'pair']]

                self.process_executed_order(order_info_df, unique_tx_id)

            elif order_info_df.at[row, 'ordertype'] == None: ### IF NOTHING IN THE ORDER_INFO_DICT, DO NOTHING
                return

            else:
                self.open_order(order_info_df, unique_tx_id)

    def loop_execute_open_order(self):

        for pair in range(len(self.pairs_price_dict)):
            self.execute_open_order(list(self.pairs_price_dict.keys())[pair],
                                    float(list(self.pairs_price_dict.values())[pair]))

    def read_test_data_to_dict(self):

        for pair in self.pairs_list:

            csv_file_path = self.ohlc_csv_test_data_folder_path + '/' + pair + '.csv'
            ohlc_data_all = pd.read_csv(csv_file_path, header=None)
            self.all_test_ohlc_df_dict[pair] = ohlc_data_all

    def read_test_data(self, pair, time_frame, last_unixtime):

        '''

        :param pair: (String) Currency pair, ex. 'BTCUSD'
        :param time_frame: (Int) Candle duration
        :param last_unixtime: (Float) Last unix time value in OHLC data table
        :param is_last_pair_and_timeframe:
        :return:
        '''


        ohlc_data_adjusted = pd.DataFrame(columns=['dtime', 'time', 'open', 'high', 'low', 'close', 'volume'])
        data_points_taken = 0

        ### FIND PREVIOUS TIME STAMP CORRESPONDING TO PREVIOUS TIME FRAME INTERVAL
        start_row = self.all_test_ohlc_df_dict[pair].iloc[(self.all_test_ohlc_df_dict[pair][0] - self.current_test_time).abs().argsort()[:1]].index[0]
        time_frame_mod_remainder = float(self.all_test_ohlc_df_dict[pair].at[start_row, 0]) % (time_frame * 60)
        prev_unix_time_stamp_for_time_frame = float(self.all_test_ohlc_df_dict[pair].at[start_row, 0]) - time_frame_mod_remainder
        next_unix_time_stamp_for_time_frame = prev_unix_time_stamp_for_time_frame + (time_frame * 60)

        ### SET STARTING ROW TO BEGIN READING AND NUMBER OF DATA POINTS DEPENDING ON WHETHER FIRST DATA POINT
        ### SET TO 720 DATA POINTS OR FULL DATA LIST (WHICHEVER IS LESS)
        loop_current_test_time = self.current_test_time

        # Used later during check to see if need to update and append past unix time interval price
        unixtime_for_past_timeframe_check = prev_unix_time_stamp_for_time_frame
        this_loop_first_unixtime = loop_current_test_time
        prev_loop_first_unixtime = loop_current_test_time - self.test_time_increment

        if last_unixtime == None:
            last_loop_unixtime = self.first_iteration_end_time
        else:
            last_loop_unixtime = self.current_test_time + self.test_time_increment

        # Determine row from ohlc data closest to next time frame increment
        row_closest_to_next_time_frame_df = self.find_row_closest_to_next_time_frame(pair, next_unix_time_stamp_for_time_frame)


        while loop_current_test_time < last_loop_unixtime:

            # Determine row from ohlc data closest to current time
            row_closest_to_current_loop_time = self.all_test_ohlc_df_dict[pair].iloc[
                (self.all_test_ohlc_df_dict[pair][0] - loop_current_test_time).abs().argsort()[:1]]
            row_closest_to_current_loop_time = row_closest_to_current_loop_time.reset_index(drop=True)

            # Used to find the "open" value for current timeframe candle
            row_closest_to_open_loop_time = self.all_test_ohlc_df_dict[pair].iloc[
                (self.all_test_ohlc_df_dict[pair][0] - prev_unix_time_stamp_for_time_frame).abs().argsort()[:1]]
            row_closest_to_open_loop_time = row_closest_to_open_loop_time.reset_index(drop=True)
            unixtime_for_row_closest_to_open_loop_time = row_closest_to_open_loop_time.iat[0, 0]

            high = self.all_test_ohlc_df_dict[pair].loc[
                self.all_test_ohlc_df_dict[pair][0].between(unixtime_for_row_closest_to_open_loop_time, loop_current_test_time)][1].max()

            low = self.all_test_ohlc_df_dict[pair].loc[
                self.all_test_ohlc_df_dict[pair][0].between(unixtime_for_row_closest_to_open_loop_time, loop_current_test_time)][1].min()

            ohlc_data_adjusted.at[data_points_taken, 'dtime'] = datetime.utcfromtimestamp(
                prev_unix_time_stamp_for_time_frame).strftime('%Y-%m-%d %H:%M:%S')
            ohlc_data_adjusted.at[data_points_taken, 'time'] = prev_unix_time_stamp_for_time_frame
            ohlc_data_adjusted.at[data_points_taken, 'open'] = row_closest_to_open_loop_time.at[0,1]
            ohlc_data_adjusted.at[data_points_taken, 'high'] = high
            ohlc_data_adjusted.at[data_points_taken, 'low'] = low
            ohlc_data_adjusted.at[data_points_taken, 'close'] = row_closest_to_current_loop_time.at[0, 1]

            if row_closest_to_next_time_frame_df.at[0,0] < loop_current_test_time:
                ohlc_data_adjusted.at[data_points_taken, 'close'] = row_closest_to_next_time_frame_df.at[0, 1]
                data_points_taken += 1
                next_unix_time_stamp_for_time_frame += (time_frame * 60)
                prev_unix_time_stamp_for_time_frame += (time_frame * 60)
                row_closest_to_next_time_frame_df = self.find_row_closest_to_next_time_frame(pair, next_unix_time_stamp_for_time_frame)

            loop_current_test_time += self.test_time_increment


        while prev_loop_first_unixtime < unixtime_for_past_timeframe_check < this_loop_first_unixtime:
            # If the timeframe unixtime interval is between previous start time and current time then need
            # to append actual timeframe interval candle close price to top of ohlc_data_adjusted

            row_closest_to_past_unixtime_df = pd.DataFrame()

            row_closest_to_past_unixtime = self.all_test_ohlc_df_dict[pair].iloc[
                (self.all_test_ohlc_df_dict[pair][0] - unixtime_for_past_timeframe_check).abs().argsort()[:1]]
            row_closest_to_past_unixtime = row_closest_to_past_unixtime.reset_index(drop=True)

            unixtime_for_row_closest_to_last_loop_time = row_closest_to_past_unixtime.iat[0, 0]

            # Used to find the "open" value for current timeframe candle
            unixtime_for_past_timeframe_check_for_last_loop = unixtime_for_past_timeframe_check - (time_frame * 60)
            row_closest_to_open_loop_time = self.all_test_ohlc_df_dict[pair].iloc[
                (self.all_test_ohlc_df_dict[pair][0] - unixtime_for_past_timeframe_check_for_last_loop).abs().argsort()[:1]]
            row_closest_to_open_loop_time = row_closest_to_open_loop_time.reset_index(drop=True)

            unixtime_for_row_closest_to_open_loop_time = row_closest_to_open_loop_time.iat[0, 0]

            high = self.all_test_ohlc_df_dict[pair].loc[
                self.all_test_ohlc_df_dict[pair][0].between(unixtime_for_row_closest_to_open_loop_time, unixtime_for_row_closest_to_last_loop_time)][1].max()

            low = self.all_test_ohlc_df_dict[pair].loc[
                self.all_test_ohlc_df_dict[pair][0].between(unixtime_for_row_closest_to_open_loop_time, unixtime_for_row_closest_to_last_loop_time)][1].min()

            row_closest_to_past_unixtime_df.at[0, 'dtime'] = datetime.utcfromtimestamp(
                unixtime_for_past_timeframe_check_for_last_loop).strftime('%Y-%m-%d %H:%M:%S')
            row_closest_to_past_unixtime_df.at[0, 'time'] = unixtime_for_past_timeframe_check_for_last_loop
            row_closest_to_past_unixtime_df.at[0, 'open'] = row_closest_to_open_loop_time.at[0,1]
            row_closest_to_past_unixtime_df.at[0, 'high'] = high
            row_closest_to_past_unixtime_df.at[0, 'low'] = low
            row_closest_to_past_unixtime_df.at[0, 'close'] = row_closest_to_past_unixtime.at[0, 1]

            ohlc_data_adjusted = pd.concat([row_closest_to_past_unixtime_df, ohlc_data_adjusted], ignore_index= True)

            unixtime_for_past_timeframe_check -= (time_frame * 60)

        return ohlc_data_adjusted, next_unix_time_stamp_for_time_frame

    def find_row_closest_to_next_time_frame(self, pair, next_unix_time_stamp_for_time_frame):

        row_closest_to_next_time_frame_df = self.all_test_ohlc_df_dict[pair].iloc[
            (self.all_test_ohlc_df_dict[pair][0] - next_unix_time_stamp_for_time_frame).abs().argsort()[:1]]
        row_closest_to_next_time_frame_df = row_closest_to_next_time_frame_df.reset_index(drop= True)

        return row_closest_to_next_time_frame_df

    def get_pair_price(self):

        # current_test_time_start_of_the_iteration = (self.current_test_time - self.test_time_increment)
        current_test_time_start_of_the_iteration = (self.current_test_time)

        for row in range(len(self.pairs_list)):

            csv_file_path = self.ohlc_csv_test_data_folder_path + '/' + self.pairs_list[row] + '.csv'
            csv_data = pd.read_csv(csv_file_path, header=None)
            row_index_closest_to_current_unixtime = csv_data.iloc[(csv_data[0] - current_test_time_start_of_the_iteration).abs().argsort()[:1]].index[0]
            price_of_row_closest_to_current_unixtime = csv_data.at[row_index_closest_to_current_unixtime, 1]
            self.pairs_price_dict[list(self.pairs_price_dict.keys())[row]] = price_of_row_closest_to_current_unixtime

        return self.pairs_price_dict

    def get_trade_balance(self):

        columns = ['eb', 'tb', 'm', 'n', 'c', 'v', 'e', 'mf']
        account_trade_balance = pd.DataFrame(columns= columns)

        # Note: 'mf' = free margin = equity - initial margin (maximum margin available to open new positions)
        # Free margin = cash + spot positions, denominated in USD
        self.free_margin = self.calc_curr_free_margin()

        # Zero some of the variables
        self.margin_amount = 0
        self.unrealized_PnL = 0
        self.open_position_cost_basis = 0
        self.open_position_current_floating_valuation = 0

        # Update 'm' = margin amount of open positions
        # Update 'n' = unrealized net profit/loss of open positions
        # Update 'c' = open position cost basis
        # Update 'v' = open position current floating valuation (simply summation of vol * current price)
        for row in range(self.open_positions_df.shape[0]):
            self.margin_amount += self.open_positions_df.at[row, 'margin']

            pair = self.open_positions_df.at[row, 'pair']
            trade_type = self.open_positions_df.at[row, 'type']
            trade_opening_price = (self.open_positions_df.at[row, 'cost'] / self.open_positions_df.at[row, 'vol'])
            trade_current_vol = (self.open_positions_df.at[row, 'vol'] - self.open_positions_df.at[row, 'vol_closed'])

            self.unrealized_PnL += utils.calculate_position_pnl(trade_type= trade_type,
                                                                trade_opening_price= trade_opening_price,
                                                                pair_current_price= self.pairs_price_dict[pair],
                                                                trade_current_vol= trade_current_vol)

            self.open_position_cost_basis += self.open_positions_df.at[row, 'cost']

            self.open_position_current_floating_valuation += (self.pairs_price_dict[pair] * trade_current_vol)

        # Update 'tb' = trade balance = free margin + margin - unrealized_pnl
        self.trade_balance = self.free_margin + self.margin_amount

        # Update 'e' = equity = trade balance + unrealized PnL
        self.equity = self.trade_balance + self.unrealized_PnL


        # Update 'ml' = margin level = (equity / initial margin) * 100
        if self.margin_amount == 0:
            self.margin_level = 0
        else:
            self.margin_level = self.equity / self.margin_amount * 100

        account_trade_balance.at[0, 'tb'] = self.trade_balance         # Update 'tb' = trade balance
        account_trade_balance.at[0, 'm'] = self.margin_amount
        account_trade_balance.at[0, 'n'] = self.unrealized_PnL
        account_trade_balance.at[0, 'c'] = self.open_position_cost_basis
        account_trade_balance.at[0, 'v'] = self.open_position_current_floating_valuation
        account_trade_balance.at[0, 'e'] = self.equity
        account_trade_balance.at[0, 'mf'] = self.free_margin
        # account_trade_balance.at[0, 'ml'] = self.margin_level

        return account_trade_balance

    def get_account_balance(self):

        '''
        Returns account cash + spot positions

        :return:
        '''

        return self.account_balance_df

    def update_cash_and_free_margin_after_trade(self, is_new_position: bool, price: float= None, volume: float=None, trade_val: float=None, fee: float= None):

        '''
        If opening or closing a trade, update cash position and free margin

        :param is_new_position:
        :param price:
        :param volume:
        :param trade_val:
        :return:
        '''

        if trade_val == None:
            trade_val = price * volume

        # Trade value gets credited or subtracted depending on whether position is being opened or closed
        if is_new_position == True:
            self.account_balance_df.at['ZUSD', 'vol'] -= trade_val # Calculate new cash position
        else:
            self.account_balance_df.at['ZUSD', 'vol'] += trade_val # Calculate closed cash position

        self.account_balance_df.at['ZUSD', 'vol'] -= fee # Subtract fee
        self.calc_curr_free_margin()  # Update account free margin balance

    def calc_curr_free_margin(self):

        '''
        Calculates current free margin (account cash + spot positions)

        :return:
        '''

        account_bal_df_temp = self.account_balance_df
        free_margin_temp = 0
        pairs_price_dict_temp = self.pairs_price_dict

        for pair in pairs_price_dict_temp:

            price = pairs_price_dict_temp[pair]
            ticker = utils.convert_pair_to_ticker(pair)

            free_margin_temp += (account_bal_df_temp.at[ticker, 'vol'] * price)

        free_margin_temp += account_bal_df_temp.at['ZUSD', 'vol']

        self.free_margin = free_margin_temp

        return self.free_margin




