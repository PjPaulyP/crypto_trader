import os
import webbrowser
import utility as utils
import pandas as pd
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash
from datetime import datetime, timedelta
import time
from dash import html
from dash import dcc
from plotly.subplots import make_subplots
import ast
import numpy as np

class trade_performance_monitoring:

    def __init__(self, trade_csv_folder_path, trade_perf_monitoring_folder_path,
                 initial_time, account_trade_balance_df, pair_price_dict, pairs_list):

        '''
        Initialize trade performance monitoring object.

        :param trade_csv_folder_path:
            (str) folder path for  trade data (open orders, open positions, closed orders)
        :param trade_perf_monitoring_folder_path:
            (str) folder path for trade performance monitoring data
        :param initial_time:
            (integer) time at initialization of main()
        :param account_trade_balance_df:
            (Pandas DataFrame) dataframe of account parameters (equity,

            eb = equivalent balance (combined balance of all currencies)
            tb = trade balance (combined balance of all equity currencies)
            m = margin amount of open positions
            n = unrealized net profit/loss of open positions
            c = cost basis of open positions
            v = current floating valuation of open positions
            e = equity = trade balance + unrealized net profit/loss
            mf = free margin = equity - initial margin (maximum margin
                available to open new positions)
            ml = margin level = (equity / initial margin) * 100

        :param pair_price_dict:
            (dict) dictionary of each ticker pair and latest prices

        '''

        # Initialize object time
        self.initial_time = initial_time
        self.current_time = self.initial_time
        self.plot_update_timeframe = 86400
        self.last_update_time = self.initial_time - (self.initial_time % self.plot_update_timeframe)

        # Initialize trade balance info
        index = account_trade_balance_df.index[0]
        self.initial_equity = account_trade_balance_df.at[index, 'e']
        self.trade_balance = 0
        self.margin_amount = 0
        self.unrealized_PnL = 0
        self.open_position_cost_basis = 0
        self.open_position_current_floating_valuation = 0
        self.equity = 0
        self.free_margin = 0
        self.margin_level = 0
        self.fees = 0
        self.trade_return_percent = 0
        self.trade_return_percent = 0

        # Current updated prices for each pair
        self.pairs_price_dict = pair_price_dict
        self.pairs_list = pairs_list

        # Portfolio KPIs
        self.return_percentage = 0
        self.trade_win_rate_list = []
        self.trade_win_rate_df = pd.DataFrame()
        self.trade_win_rate_rolling_df = pd.DataFrame(columns= ['time', 'rolling_win_rate'])
        self.trade_win_rate_rolling_avg_period = 7
        self.total_num_trades = 0

        # Data for the portfolio value vs. time chart
        self.portfolio_value_df = pd.DataFrame(columns= ['time', 'portfolio_value'])
        self.portfolio_value_df_raw = pd.DataFrame(columns= ['time', 'portfolio_value'])
        self.time_list = [self.last_update_time] # Initialize time_list with unixtime of initial date @ 00:00
        self.portfolio_value_df.at[0, 'time'] = self.initial_time
        self.portfolio_value_df.at[0, 'portfolio_value'] = self.initial_equity

        # Data for the open position table
        self._open_positions_table_header = ['pair', 'type', 'open_volume', 'entry_price', 'current_price', 'leverage', 'pnl']
        self.open_positions_table_df = pd.DataFrame(columns= self._open_positions_table_header)

        # File / Folder Paths
        self.trade_csv_folder_path = trade_csv_folder_path
        self.trade_csv_file_names = ['open_orders', 'closed_orders', 'open_positions']
        self.trade_perf_monitoring_folder_path = trade_perf_monitoring_folder_path

        # Tade data dataframes
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

        spot_open_positions_header = ['ticker', 'vol', 'entry_price']

        self.open_orders_df = pd.DataFrame(columns= open_orders_headers)
        self.closed_orders_df_edited = pd.DataFrame(columns= closed_orders_header)
        self.open_positions_df = pd.DataFrame(columns= open_positions_header)
        self.open_spot_positions = pd.DataFrame(columns= spot_open_positions_header)

        self.update_trade_balance(account_trade_balance_df, self.open_spot_positions, initial_time)

    def update_pairs_price_dict(self, pairs_price_dict):

        '''
        Update pair pricing dictionary for object variable

        :param pairs_price_dict:
            (dict) dictionary of each crypto-currency pair with updated prices
        :return:
            self.pairs_price_dict = same as input pairs_price_dict
        '''

        self.pairs_price_dict = pairs_price_dict

        return self.pairs_price_dict

    def update_trade_balance(self, account_trade_balance_df, open_spot_positions, current_time):

        self.current_time = current_time

        index = account_trade_balance_df.index[0]

        self.trade_balance = account_trade_balance_df.at[index, 'tb']
        self.margin_amount = account_trade_balance_df.at[index, 'm']
        self.unrealized_PnL = account_trade_balance_df.at[index, 'n']
        self.open_position_cost_basis = account_trade_balance_df.at[index, 'c']
        self.open_position_current_floating_valuation = account_trade_balance_df.at[index, 'v']
        self.equity = account_trade_balance_df.at[index, 'e']
        self.free_margin = account_trade_balance_df.at[index, 'mf']
        # self.margin_level = account_trade_balance_df.at[index, 'ml']

        # print('4.0 EQUITY TROUBLESHOOTING ---------------------')
        # print(self.current_time, self.equity)
        # print('4.1 EQUITY TROUBLESHOOTING ---------------------')
        # print(self.free_margin, self.margin_amount)
        # print('4.2 EQUITY TROUBLESHOOTING ---------------------')
        # print(self.unrealized_PnL, self.trade_balance)

        # Update unrealized pnl to account for spot positions, which don't show on the account_trade_balance df
        for row in range(open_spot_positions.shape[0]):
            pair = utils.convert_ticker_to_pair(open_spot_positions.at[row, 'ticker'])
            entry_price = open_spot_positions.at[row, 'entry_price']
            pair_current_price = self.pairs_price_dict[pair]
            trade_current_vol = open_spot_positions.at[row, 'vol']

            self.unrealized_PnL += utils.calculate_position_pnl(trade_type= 'buy',
                                                                trade_opening_price= entry_price,
                                                                pair_current_price= pair_current_price,
                                                                trade_current_vol= trade_current_vol)

        # print('4.3 EQUITY TROUBLESHOOTING ---------------------')
        # print(self.unrealized_PnL)

        if self.portfolio_value_df.shape[0] == 0:
            row_port_val_df = 0
        else:
            row_port_val_df = self.portfolio_value_df.shape[0]

        self.portfolio_value_df_raw.at[row_port_val_df, 'time'] = self.current_time
        self.portfolio_value_df_raw.at[row_port_val_df, 'portfolio_value'] = account_trade_balance_df.at[index, 'e']

    def update_time_list(self):

        time_delta = self.current_time - self.last_update_time
        time_delta_fraction = time_delta / self.plot_update_timeframe

        # Append discrete time data points from last update time to current time
        while time_delta_fraction >= 1:

            if self.last_update_time not in self.time_list:
                self.time_list.append(self.last_update_time)
            self.last_update_time += self.plot_update_timeframe
            time_delta = self.current_time - self.last_update_time
            time_delta_fraction = time_delta / self.plot_update_timeframe

        # Append time for "today's" to end of time list
        if self.last_update_time not in self.time_list and self.current_time >= self.last_update_time:
            self.time_list.append(self.last_update_time)

        return self.time_list

    def update_portfolio_value_frame(self):

        self.portfolio_value_df = pd.DataFrame(columns= ['time', 'portfolio_value'])

        for row in range(len(self.time_list)):

            time = datetime.utcfromtimestamp(self.time_list[row]).strftime('%Y-%m-%d')
            self.portfolio_value_df.at[row, 'time'] = time

            row_closest_to_loop_time = self.portfolio_value_df_raw.iloc[
                (self.portfolio_value_df_raw['time'] - self.time_list[row]).abs().argsort()[:1]]
            row_closest_to_loop_time = row_closest_to_loop_time.reset_index(drop=True)
            self.portfolio_value_df.at[row, 'portfolio_value'] = row_closest_to_loop_time.at[0, 'portfolio_value']

        # Make sure last portfolio value is latest value for the "today" value
        row_closest_to_current_time = self.portfolio_value_df_raw.iloc[
            (self.portfolio_value_df_raw['time'] - self.current_time).abs().argsort()[:1]]
        row_closest_to_current_time  = row_closest_to_current_time.reset_index(drop=True)
        self.portfolio_value_df.at[self.portfolio_value_df.shape[0] - 1, 'portfolio_value'] = row_closest_to_current_time.at[0, 'portfolio_value']

    def update_trade_win_rate_frame(self):

        # First update self.trade_win_rate_df (raw list without rolling average applied)

        self.update_trade_win_rate()

        closed_orders_df_edited_temp = self.closed_orders_df_edited.reset_index(drop= True)

        trade_win_rate_df = pd.DataFrame()
        time_list = []

        for i in range(len(self.time_list)):
            time_list.append(datetime.utcfromtimestamp(self.time_list[i]).strftime('%Y-%m-%d'))

        trade_win_rate_df['time'] = time_list
        trade_win_rate_df['win_rate'] = self.trade_win_rate_list
        self.trade_win_rate_df = trade_win_rate_df

        # Now update self.trade_win_rate_rolling_df. Essentially, return the average trade win rate
        # over the last N days
        if self.trade_win_rate_df.shape[0] >= self.trade_win_rate_rolling_avg_period:

            # Create temp df to append the rolling trade data info to
            temp_df = pd.DataFrame()
            temp_df_row = 0

            # win_rate_roll_avg = utils.apply_rolling_average_list(self.trade_win_rate_df.win_rate.to_list(), self.trade_win_rate_rolling_avg_period)

            temp_df['time'] = self.trade_win_rate_df.time.to_list()[self.trade_win_rate_rolling_avg_period - 1:]

            for row_a in range(self.trade_win_rate_rolling_avg_period - 1, len(self.trade_win_rate_df.time.to_list())):

                win_rate_rolling_avg = 0
                num_trades = 0
                number_of_positive_trades = 0

                row_time = self.trade_win_rate_df.at[row_a, 'time']
                row_time_unix = time.mktime(datetime.strptime(row_time, "%Y-%m-%d").timetuple())

                # Subtract N number of days in seconds (86,400 seconds in a day)
                time_lower_thresh_unix = row_time_unix - (self.trade_win_rate_rolling_avg_period * 86400)

                for row_b in range(closed_orders_df_edited_temp.shape[0]):
                    if time_lower_thresh_unix <= float(closed_orders_df_edited_temp.at[row_b, 'last_closetm']) <= row_time_unix:

                        if closed_orders_df_edited_temp.at[row_b, 'pnl'] > 0:
                            number_of_positive_trades += 1
                        num_trades += 1

                if num_trades == 0:
                    win_rate_rolling_avg = 0
                else:
                    win_rate_rolling_avg = number_of_positive_trades / num_trades * 100

                temp_df.at[temp_df_row, 'rolling_win_rate'] = win_rate_rolling_avg
                temp_df_row += 1

            self.trade_win_rate_rolling_df = temp_df

    def update_trade_win_rate(self):

        updated_closed_trades_edited_df = utils.read_csv_data(self.trade_perf_monitoring_folder_path, 'closed_orders_edited').reset_index()
        updated_closed_trades_edited_df.rename(columns={'Unnamed: 0': 'ordertxid'}, inplace=True)
        number_of_positive_trades = 0

        for i in range(updated_closed_trades_edited_df.shape[0]):
            if updated_closed_trades_edited_df.at[i, 'pnl'] > 0:
                number_of_positive_trades += 1

        if len(self.time_list) > len(self.trade_win_rate_list):

            if updated_closed_trades_edited_df.shape[0] != 0:
                self.trade_win_rate_list.append(number_of_positive_trades / updated_closed_trades_edited_df.shape[0] * 100)
            else:
                self.trade_win_rate_list.append(0)

    def update_open_positions_frame(self):

        self.open_positions_df = utils.read_csv_data(self.trade_csv_folder_path, 'open_positions')
        self.open_spot_positions_df = utils.read_csv_data(self.trade_csv_folder_path, 'open_spot_positions')
        self.open_spot_positions_df = self.open_spot_positions_df.drop(columns= ['Unnamed: 0'])
        self.open_spot_positions_df = self.open_spot_positions_df[self.open_spot_positions_df.vol != 0].reset_index(drop= True)

        self.open_positions_table_df = pd.DataFrame(columns= self._open_positions_table_header)

        for row in range(self.open_positions_df.shape[0]):

            table_row = self.open_positions_table_df.shape[0]

            current_price = self.pairs_price_dict[self.open_positions_df.at[row, 'pair']]

            self.open_positions_table_df.at[table_row , 'pair'] = self.open_positions_df.at[row, 'pair']
            self.open_positions_table_df.at[table_row , 'type'] = self.open_positions_df.at[row, 'type']
            self.open_positions_table_df.at[table_row , 'open_volume'] = (self.open_positions_df.at[row, 'vol'] - self.open_positions_df.at[row, 'vol_closed']).round(2)
            self.open_positions_table_df.at[table_row , 'entry_price'] = \
                (self.open_positions_df.at[row, 'cost'] / self.open_positions_df.at[row, 'vol'])
            self.open_positions_table_df.at[table_row , 'current_price'] = current_price
            self.open_positions_table_df.at[table_row , 'leverage'] = str(round(self.open_positions_df.at[row, 'cost'] / self.open_positions_df.at[row, 'margin'])) + ':1'
            self.open_positions_table_df.at[table_row , 'pnl'] = (utils.calculate_position_pnl(trade_type= self.open_positions_table_df.at[row, 'type'],
                                                                                        trade_opening_price= self.open_positions_table_df.at[row, 'entry_price'],
                                                                                        pair_current_price= self.open_positions_table_df.at[row, 'current_price'],
                                                                                        trade_current_vol= self.open_positions_table_df.at[row, 'open_volume'])).round(2)

        for row in range(self.open_spot_positions_df.shape[0]):

            table_row = self.open_positions_table_df.shape[0]

            ticker = self.open_spot_positions_df.at[row, 'ticker']
            pair = utils.convert_ticker_to_pair(ticker)
            current_price = self.pairs_price_dict[pair]
            volume = self.open_spot_positions_df.at[row, 'vol']

            self.open_positions_table_df.at[table_row, 'pair'] = pair
            self.open_positions_table_df.at[table_row, 'type'] = 'buy'
            self.open_positions_table_df.at[table_row, 'open_volume'] = volume
            self.open_positions_table_df.at[table_row, 'entry_price'] = self.open_spot_positions_df.at[row, 'entry_price']
            self.open_positions_table_df.at[table_row, 'current_price'] = current_price
            self.open_positions_table_df.at[table_row, 'leverage'] = '1:1'
            self.open_positions_table_df.at[table_row, 'pnl'] = (
                utils.calculate_position_pnl(trade_type=self.open_positions_table_df.at[row, 'type'],
                                             trade_opening_price=self.open_positions_table_df.at[row, 'entry_price'],
                                             pair_current_price=self.open_positions_table_df.at[row, 'current_price'],
                                             trade_current_vol=self.open_positions_table_df.at[row, 'open_volume'])).round(2)

    def loop_trade_perf_functions(self, account_trade_balance_df, open_spot_positions, current_time, pairs_price_dict):

        self.update_pairs_price_dict(pairs_price_dict)
        self.update_trade_balance(account_trade_balance_df, open_spot_positions, current_time)
        self.create_closed_orders_df_edited()
        self.update_time_list()
        self.update_portfolio_value_frame()
        self.update_trade_win_rate_frame()
        self.update_open_positions_frame()
        self.update_total_num_trades()
        self.update_trade_return_percent()
        self.update_trade_return_percent()

    def update_total_num_trades(self):

        updated_closed_trades_df = utils.read_csv_data(self.trade_csv_folder_path, 'closed_orders').reset_index(drop= True)
        updated_closed_trades_df = updated_closed_trades_df[updated_closed_trades_df.status != 'canceled']
        self.total_num_trades = updated_closed_trades_df.shape[0]

    def update_trade_return_percent(self):

        self.trade_return_percent = ((self.equity / self. initial_equity) - 1) * 100

    def update_trade_return_percent(self):

        closed_orders_edited_df_temp = self.closed_orders_df_edited
        total_cost_traded = closed_orders_edited_df_temp.cost.sum()
        trade_return_percent = (closed_orders_edited_df_temp.cost * closed_orders_edited_df_temp.return_percent / total_cost_traded).sum()
        self.trade_return_percent = trade_return_percent

        return trade_return_percent

    def create_closed_orders_df_edited(self):

        closed_orders_df = utils.read_csv_data(self.trade_csv_folder_path, 'closed_orders')
        closed_orders_df.rename(columns={'Unnamed: 0': 'ordertxid'}, inplace=True)
        closed_orders_df = utils.reverse_df_order(closed_orders_df).reset_index(drop= True) # Flip order, since earliest orders on raw df start at bottom
        closed_orders_df['is_original_order'] = True # Original orders are new orders that open new position (i.e., orders that don't close existing)
        closed_orders_df['close_price'] = None
        closed_orders_df['total_fees'] = 0
        closed_orders_df['pnl'] = None
        closed_orders_df['return_percent'] = None
        closed_orders_df['closing_order_txid'] = None
        closed_orders_df['last_closetm'] = None
        closed_orders_df['opentm_date'] = None
        closed_orders_df['closetm_date'] = None
        closed_orders_df['is_stop_lossed'] = None
        closed_orders_df['position_duration [Hr]'] = None


        closed_orders_df_temp = pd.DataFrame(columns= closed_orders_df.columns.to_list())
        pairs_aggregate_vol_dict = {}.fromkeys(self.pairs_list, 0)


        for row in range(closed_orders_df.shape[0]):

            # Create temp df to house orders that found to close the original order
            sub_orders_df_temp = pd.DataFrame()

            pair = closed_orders_df.at[row, 'descr_pair']
            is_original_order = closed_orders_df.at[row, 'is_original_order']
            order_status = closed_orders_df.at[row, 'status']


            if pairs_aggregate_vol_dict[pair] == 0 and\
                    order_status == 'closed' and \
                    is_original_order == True:

                ordertype = closed_orders_df.at[row, 'descr_type']
                cost = closed_orders_df.at[row, 'cost']
                total_fees = closed_orders_df.at[row, 'fee']
                vol_exec = closed_orders_df.at[row, 'vol_exec']
                opentm = closed_orders_df.at[row, 'opentm']
                opentm_date = datetime.utcfromtimestamp(opentm).strftime('%Y-%m-%d %H:%M:%S')

                # Set corresponding dict pair value to volume of original order row
                if ordertype == 'buy':
                    pairs_aggregate_vol_dict[pair] += vol_exec
                else:
                    pairs_aggregate_vol_dict[pair] -= vol_exec

                # Now scan through df for the orders that closed the original order
                for sub_row in range(row, closed_orders_df.shape[0]):

                    sub_pair = closed_orders_df.at[sub_row, 'descr_pair']
                    sub_ordertype = closed_orders_df.at[sub_row, 'descr_type']
                    sub_vol_exec = closed_orders_df.at[sub_row, 'vol_exec']
                    sub_order_status = closed_orders_df.at[sub_row, 'status']

                    if pair == sub_pair and \
                            sub_order_status == 'closed' and \
                            ordertype != sub_ordertype:

                        # If row is an order to close original, then set to False
                        closed_orders_df.at[sub_row, 'is_original_order'] = False

                        # If row is not original order (i.e., order to close existing position), then add to temp df
                        this_sub_row = closed_orders_df.iloc[sub_row].to_frame().transpose()
                        sub_orders_df_temp = pd.concat([sub_orders_df_temp, this_sub_row], ignore_index=True).reset_index(drop= True)

                        if sub_ordertype == 'buy':
                            pairs_aggregate_vol_dict[pair] += sub_vol_exec
                        else:
                            pairs_aggregate_vol_dict[pair] -= sub_vol_exec


                        if pairs_aggregate_vol_dict[pair] == 0:

                            close_price = ((sub_orders_df_temp.vol_exec * sub_orders_df_temp.price) / sub_orders_df_temp.vol_exec.sum()).sum()
                            total_fees += sub_orders_df_temp.fee.sum()

                            if ordertype == 'buy':
                                pnl = sub_orders_df_temp.cost.sum() - cost - total_fees

                            elif ordertype == 'sell':
                                pnl = cost - sub_orders_df_temp.cost.sum() - total_fees

                            return_percent = pnl / cost * 100

                            close_tx_id_list = sub_orders_df_temp.ordertxid.to_list()
                            close_tx_id_str = ','.join(close_tx_id_list)

                            last_closetm = max(sub_orders_df_temp.closetm.to_list()) # Time of the last order that closed this existing position
                            closetm_date = datetime.utcfromtimestamp(last_closetm).strftime('%Y-%m-%d %H:%M:%S')

                            closed_orders_df.at[row, 'close_price'] = close_price
                            closed_orders_df.at[row, 'total_fees'] = total_fees
                            closed_orders_df.at[row, 'pnl'] = pnl
                            closed_orders_df.at[row, 'return_percent'] = return_percent
                            closed_orders_df.at[row, 'closing_order_txid'] = close_tx_id_str
                            closed_orders_df.at[row, 'last_closetm'] = last_closetm
                            closed_orders_df.at[row, 'opentm_date'] = opentm_date
                            closed_orders_df.at[row, 'closetm_date'] = closetm_date
                            closed_orders_df.at[row, 'position_duration [Hr]'] = (last_closetm - opentm) / 60 / 60 # Turn seconds to hours

                            if 'stop-loss' in sub_orders_df_temp.descr_ordertype.to_list():
                                closed_orders_df.at[row, 'is_stop_lossed'] = True
                            else:
                                closed_orders_df.at[row, 'is_stop_lossed'] = False

                            # If row is an original order (i.e., to open new position), then add to temp df
                            this_row = closed_orders_df.iloc[row].to_frame().transpose()
                            closed_orders_df_temp = pd.concat([closed_orders_df_temp, this_row],
                                                              ignore_index=True).reset_index(drop=True)

                            break


        closed_orders_df_temp = utils.reverse_df_order(closed_orders_df_temp)
        closed_orders_df_temp.index = closed_orders_df_temp.ordertxid
        closed_orders_df_temp = closed_orders_df_temp.drop(columns= ['ordertxid'])
        self.closed_orders_df_edited = closed_orders_df_temp

        # Send to csv
        file_path = self.trade_perf_monitoring_folder_path + '/' + 'closed_orders_edited' + '.csv'
        closed_orders_df_temp.to_csv(file_path)

class portfolio_visuals:

    _time_list = []
    _portfolio_value_list = []

    _chart_theme = 'plotly_dark'
    _app = Dash(external_stylesheets=[dbc.themes.DARKLY])
    _app_tabs = html.Div()

    _open_position_table_headers = ['Pairs', 'Type', 'Open Vol', 'Entry Price', 'Current Price', 'Leverage', ' PnL']
    _open_position_df_headers = ['pair', 'type', 'open_volume', 'entry_price', 'current_price', 'leverage',
                                 'pnl']
    _open_position_table_df = pd.DataFrame(columns= _open_position_df_headers)

    _trade_win_rate_df = pd.DataFrame()
    _trade_win_rate_rolling_df = pd.DataFrame(columns= ['time', 'rolling_win_rate'])
    _trade_pnl_list = []
    _total_num_trades = 0
    _n_intervals_seconds = 5
    _trade_return_percent = 0
    _unrealized_PnL = 0


    def __init__(self, initial_time, initial_equity: float, pairs: list, timeframes: list,
                 ohlc_folder_name: str, trade_performance_monitoring_folder_name: str, portfolio_val_list_file_name: str, portfolio_value_time_list_file_name: str,
                 main_to_dash_bridge_obj = None):

        self._ohlc_folder_name = ohlc_folder_name
        self._trade_performance_monitoring_folder_name = trade_performance_monitoring_folder_name
        self._portfolio_val_list_file_name = portfolio_val_list_file_name
        self._portfolio_value_time_list_file_name = portfolio_value_time_list_file_name


        self.main_to_dash_bridge_obj = main_to_dash_bridge_obj
        self._app_tabs = self.create_app_tabs()

        self._initial_time = initial_time
        self._initial_time_data = datetime.utcfromtimestamp(int(self._initial_time)).strftime('%B %d, %Y %H:%M:%S')
        self._initial_equity = initial_equity

        self._current_time = initial_time
        self._portfolio_value_list.append(initial_equity)

        self._pairs = pairs
        self._timeframes = timeframes

        self.create_app_layout()
        if self._app is not None and hasattr(self, "callbacks"):
            self.callbacks(self._app)


    def create_app_tabs(self):

        app_tabs = html.Div(
            [
                dbc.Tabs(
                    [
                        dbc.Tab(label='Portfolio Summary', tab_id='tab-portfolio-summary'),
                        dbc.Tab(label='Candlestick Charts', tab_id='tab-candlestick-charts')
                    ],
                    id='tabs',
                    active_tab='tab-portfolio-summary'
                )
            ]
        )

        return app_tabs

    def create_app_layout(self):

        self._app.layout = dbc.Container(

            [
                dbc.Row(dbc.Col(self._app_tabs, width=12)),
                html.Div(id='content', children=[])
            ], fluid=True
        )

    def callbacks(self, _app):

        @_app.callback(
            Output('content', 'children'),
            Input('tabs', 'active_tab')
        )
        def switch_tab(tab_chosen):
            if tab_chosen == 'tab-portfolio-summary':
                return self.create_portfolio_summary_layout()
            elif tab_chosen == 'tab-candlestick-charts':
                return self.create_candlestick_charts_layout()
            return html.P("This shouldn't be displayed for now")

        @_app.callback(
            Output('initial-time', 'children'),
            Output('current-time', 'children'),
            [Input('update-current-time', 'n_intervals')]
        )
        def update_portfolio_indicators(n_intervals):

            initial_time, initial_cash, pairs, timeframes, \
            current_time, time_list, portfolio_value_list, open_positions_table_df, \
            trade_win_rate_df, trade_win_rate_rolling_df, total_num_trades, trade_return_percent, \
            unrealized_PnL = self.main_to_dash_bridge_obj.read_visual_data_from_csv()

            self.update_portfolio_visuals_params(initial_time, initial_cash, pairs, timeframes,
                                                 current_time, time_list, portfolio_value_list,
                                                 open_positions_table_df, trade_win_rate_df, trade_win_rate_rolling_df,
                                                 total_num_trades, trade_return_percent, unrealized_PnL)

            if current_time == '0':
                current_time_data = 'Initializing application...'
            else:
                current_time_data = datetime.utcfromtimestamp(int(current_time)).strftime('%B %d, %Y %H:%M:%S')

            if initial_time == '0':
                initial_time_data = 'Initializing application...'
            else:
                initial_time_data = datetime.utcfromtimestamp(int(initial_time)).strftime('%B %d, %Y %H:%M:%S')

            return 'Initial Time: ' + initial_time_data + ' UTC', \
                   'Current Time: ' + current_time_data + ' UTC'

        @_app.callback(
            Output('portfolio-indicators', 'figure'),
            [Input('update-portfolio-indicators', 'n_intervals')]
        )
        def update_portfolio_indicators(n_intervals):

            equity = round(self._portfolio_value_list[-1], 2)

            equity_data = go.Indicator(
                title={'text': "<span style= 'font-size:1em;color:white'>Equity</span>"},
                domain={'row': 0, 'column': 0},
                value=equity,
                mode="number",
                number={'valueformat': 'f', 'font': {'size': 40}, 'prefix': '$', 'suffix': ' USD'}
            )

            trade_return_percent = round(self._trade_return_percent, 2)

            trade_return_percent_data = go.Indicator(
                title={'text': "<span style= 'font-size:1em;color:white'>Trade Return</span>"},
                domain={'row': 0, 'column': 1},
                value=trade_return_percent,
                mode="number",
                number={'font': {'size': 40}, 'suffix': '%'}
            )

            if self._unrealized_PnL < 0:
                unrealized_PnL_sign = '-'
            else:
                unrealized_PnL_sign = ''

            unrealized_PnL = round(abs(self._unrealized_PnL), 2)

            unrealized_PnL_data = go.Indicator(
                title={'text': "<span style= 'font-size:1em;color:white'>Unrealized PnL</span>"},
                domain={'row': 0, 'column': 2},
                value=unrealized_PnL,
                mode="number",
                number={'font': {'size': 40}, 'prefix': unrealized_PnL_sign + '$', 'suffix': ' USD'}
            )

            if self._trade_win_rate_df.shape[0] == 0:
                win_rate_val = 100
            else:
                win_rate_val = self._trade_win_rate_df.at[self._trade_win_rate_df.shape[0] - 1, 'win_rate']

            win_rate_data = go.Indicator(
                title={'text': "<span style= 'font-size:1em;color:white'>Win Rate</span>"},
                domain={'row': 0, 'column': 3},
                value=win_rate_val,
                mode="number",
                number={'font': {'size': 40}, 'suffix': '%'}
            )

            total_num_trades_data = go.Indicator(
                title={'text': "<span style= 'font-size:1em;color:white'>Total Trade Count</span>"},
                domain={'row': 0, 'column': 4},
                value=self._total_num_trades,
                mode="number",
                number={'font': {'size': 40}}
            )

            return {
                "data": [equity_data, trade_return_percent_data, unrealized_PnL_data, win_rate_data,
                         total_num_trades_data],
                "layout": go.Layout(
                    template=self._chart_theme,
                    margin=dict(t=100),
                    grid={'rows': 1, 'columns': 5})
            }

        @_app.callback(
            Output('portfolio-value-chart', 'figure'),
            [Input('update-portfolio-value-chart', 'n_intervals')]
        )
        def update_portfolio_value_chart(n_intervals):

            data = go.Scatter(
                x=list(self._time_list), y=list(self._portfolio_value_list),
                mode='lines', name='Portfolio Value')

            if len(self._time_list) != 0:
                x_range_max = datetime.strftime(datetime.strptime(max(self._time_list), "%Y-%m-%d") + timedelta(days=1),
                                                "%Y-%m-%d")
                x_range = [min(self._time_list), x_range_max]
            else:
                x_range = [0, 1]
            if len(self._portfolio_value_list) != 0:

                y_range_min = utils.round_nearest_hundred(min(self._portfolio_value_list), min_or_max='min')
                y_range_max = utils.round_nearest_hundred(max(self._portfolio_value_list), min_or_max='max')

                y_range = [y_range_min, y_range_max]

            else:
                y_range = [0, 1]

            return {
                "data": [data],
                "layout": go.Layout(
                    template=self._chart_theme,
                    height=400,
                    margin=dict(t=50, b=50, l=25, r=25),
                    xaxis_tickfont_size=12,
                    xaxis=dict(
                        range=x_range),
                    yaxis=dict(
                        range=y_range,
                        title='Value [$ USD]',
                        titlefont_size=14,
                        tickfont_size=12
                    )
                )
            }

        @_app.callback(
            Output('win-rate-chart', 'figure'),
            [Input('update-win-rate-chart', 'n_intervals')]
        )
        def update_win_rate_chart(n_intervals):

            if self._trade_win_rate_df.shape[0] == 0:
                trade_win_rate_time = [self._current_time]
                trade_win_rate_val = ['N/A']
            else:
                trade_win_rate_time = self._trade_win_rate_df.time.to_list()
                trade_win_rate_val = self._trade_win_rate_df.win_rate.round(2).to_list()

            rolling_trade_win_rate_time = self._trade_win_rate_rolling_df.time.to_list()
            rolling_trade_win_rate_val = self._trade_win_rate_rolling_df.rolling_win_rate.round(2).to_list()

            trade_win_rate_data = go.Scatter(
                x=list(trade_win_rate_time), y=list(trade_win_rate_val),
                mode='lines', name='Win Rate')

            rolling_trade_win_rate_data = go.Scatter(
                x=list(rolling_trade_win_rate_time), y=list(rolling_trade_win_rate_val),
                mode='lines', name='Win Rate (Rolling 7 Day)')

            if (self._trade_win_rate_df.shape[0]) != 0:
                x_range_max = datetime.strftime(datetime.strptime(max(self._time_list), "%Y-%m-%d") + timedelta(days=1),
                                                "%Y-%m-%d")
                x_range = [min(self._trade_win_rate_df.time), x_range_max]
            else:
                x_range = [0, 1]

            return {
                "data": [trade_win_rate_data, rolling_trade_win_rate_data],
                "layout": go.Layout(
                    template=self._chart_theme,
                    height=400,
                    margin=dict(t=50, b=50, l=25, r=25),
                    xaxis_tickfont_size=12,
                    xaxis=dict(
                        range=x_range),
                    yaxis=dict(
                        range=[0, 100],
                        title='[%]',
                        titlefont_size=14,
                        tickfont_size=12
                    ),
                    legend=dict(
                        yanchor='top',
                        xanchor='right',
                        y=0.99,
                        x=0.99
                    )
                )
            }

        @_app.callback(
            Output('open-positions-table', 'figure'),
            [Input('update-open-positions-table', 'n_intervals')]
        )
        def update_open_positions_table(n):

            data = go.Table(
                header=dict(values=self._open_position_table_headers),
                cells=dict(values=[self._open_position_table_df.pair.to_list(),
                                   self._open_position_table_df.type.to_list(),
                                   self._open_position_table_df.open_volume.to_list(),
                                   self._open_position_table_df.entry_price.to_list(),
                                   self._open_position_table_df.current_price.to_list(),
                                   self._open_position_table_df.leverage.to_list(),
                                   self._open_position_table_df.pnl.to_list()]))

            return {
                "data": [data],
                "layout": go.Layout(
                    template=self._chart_theme,
                    margin={"t": 10, "l": 10}
                )
            }

        @_app.callback(
            Output(component_id='candlestick_chart', component_property='figure'),
            Input(component_id='pair', component_property='value'),
            Input(component_id='timeframes', component_property='value'),
            Input(component_id='candlestick_chart', component_property='relayoutData')
        )
        def build_candlestick_chart(pair, timeframe, relayoutdata):
            ohlc_folder = self._ohlc_folder_name
            ohlc_filename = pair + '_' + str(timeframe)
            ohlc_chart_name = pair + ' ' + str(timeframe)
            trade_performance_monitoring_folder = self._trade_performance_monitoring_folder_name
            portfolio_val_list_filename = self._portfolio_val_list_file_name
            portfolio_value_time_list_filename = self._portfolio_value_time_list_file_name

            ohlc_df = utils.read_csv_data(csv_folder_name=ohlc_folder, csv_file_name=ohlc_filename)

            portfolio_val_list = utils.read_csv_data(csv_folder_name=trade_performance_monitoring_folder,
                                                     csv_file_name=portfolio_val_list_filename)
            portfolio_value_time_list = utils.read_csv_data(csv_folder_name=trade_performance_monitoring_folder,
                                                            csv_file_name=portfolio_value_time_list_filename)
            portfolio_val_df = pd.DataFrame(data={'dtime': portfolio_value_time_list.col1.to_list(),
                                                  'portfolio_val': portfolio_val_list.col1.to_list()})

            fig = make_subplots(rows=3,
                                cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.04,
                                subplot_titles=('PORTFOLIO VALUE', ohlc_chart_name, 'EMA DERIVATIVE'),
                                specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]]
                                )
            # ==================================================================================================================== #
            # CHART #1 - PORTFOLIO VALUE CHART

            fig.append_trace(go.Scatter(
                x=portfolio_value_time_list.col1.to_list(),
                y=portfolio_val_list.col1.to_list(),
                mode='lines',
                name='Portfolio Value'
            ), row=1, col=1)

            # ==================================================================================================================== #
            # CHART #2 - CANDLESTICK CHART

            fig.append_trace(go.Candlestick(
                open=ohlc_df.open,
                high=ohlc_df.high,
                low=ohlc_df.low,
                close=ohlc_df.close,
                x=ohlc_df.dtime,
                name=ohlc_chart_name
            ), row=2, col=1)

            # ==================================================================================================================== #
            # CHART #3 - EMA DERIVATIVE

            fig.append_trace(go.Scatter(x=ohlc_df.dtime,
                                        y=ohlc_df.ema_12_derivative,
                                        mode='lines',
                                        name='EMA_12_d',
                                        # visible= 'legendonly'
                                        ), row=3, col=1)

            y_zero = [0] * ohlc_df.shape[0]
            zero_line = go.Scatter(x=ohlc_df.dtime,
                                   y=y_zero,
                                   mode='lines',
                                   name='ZERO',
                                   # visible= 'legendonly'
                                   )
            fig.add_trace(zero_line, row=3, col=1)

            # ==================================================================================================================== #
            # DISPLAY TRADES

            closed_orders_edited_filename = 'closed_orders_edited'

            trades_df = utils.read_csv_data(csv_folder_name=trade_performance_monitoring_folder,
                                            csv_file_name=closed_orders_edited_filename)

            trades_pair_df = trades_df[trades_df.descr_pair == pair].loc[:, [
                                                                                'ordertxid',
                                                                                'opentm_date',
                                                                                'price',
                                                                                'vol',
                                                                                'close_price',
                                                                                'pnl',
                                                                                'return_percent',
                                                                                'is_stop_lossed',
                                                                                'descr_type',
                                                                                'closetm_date'
                                                                            ]].reset_index(drop=True)

            trades_pair_df['marker_type'] = np.where(trades_pair_df['descr_type'] == 'buy', 'triangle-up', 'triangle-down')
            trades_pair_df['marker_color'] = np.where(trades_pair_df['pnl'] > 0, 'lime', 'magenta')

            trade_trace = go.Scatter(x=trades_pair_df.opentm_date,
                                     y=trades_pair_df.price,
                                     text=trades_pair_df.ordertxid,
                                     customdata=trades_pair_df,
                                     hovertemplate="<br>".join([
                                         "Open Date: %{x}",
                                         "ordertxid: %{text}",
                                         "Open Price: $%{y}",
                                         "Close Price: $%{customdata[4]}",
                                         "pnl: $%{customdata[5]}",
                                         "return_percent: %%{customdata[6]}",
                                         "Close Date: %{customdata[9]}",
                                         "is_stop_lossed: %{customdata[7]}"
                                     ]),
                                     mode='markers',
                                     name='Trades',
                                     marker=go.Marker(
                                         size=15,
                                         symbol=trades_pair_df.marker_type,
                                         color=trades_pair_df.marker_color),
                                     # visible='legendonly'
                                     )

            fig.append_trace(trade_trace, row=2, col=1)

            # ==================================================================================================================== #
            # DISPLAY CLOSE TRADES

            close_trade_trace = go.Scatter(x=trades_pair_df.closetm_date,
                                           y=trades_pair_df.close_price,
                                           text=trades_pair_df.ordertxid,
                                           customdata=trades_pair_df,
                                           hovertemplate="<br>".join([
                                               "Open Date: %{x}",
                                               "ordertxid: %{text}",
                                               "Open Price: $%{customdata[2]}",
                                               "Close Price: $%{y}",
                                               "pnl: $%{customdata[5]}",
                                               "return_percent: %%{customdata[6]}",
                                               "is_stop_lossed: %{customdata[7]}"
                                           ]),
                                           mode='markers',
                                           name='Closing Trades',
                                           marker=go.Marker(
                                               size=10,
                                               symbol='star',
                                               color='yellow'),
                                           # visible= 'legendonly'
                                           )

            fig.add_trace(close_trade_trace, row=2, col=1)

            # ==================================================================================================================== #
            # EMA LINES

            EMA_12_trace = go.Scatter(x=ohlc_df.dtime,
                                      y=ohlc_df.EMA_12,
                                      mode='lines',
                                      name='EMA_12',
                                      visible='legendonly'
                                      )

            EMA_21_trace = go.Scatter(x=ohlc_df.dtime,
                                      y=ohlc_df.EMA_21,
                                      mode='lines',
                                      name='EMA_21',
                                      visible='legendonly'
                                      )

            fig.add_trace(EMA_12_trace, secondary_y=False, row=2, col=1)
            fig.add_trace(EMA_21_trace, secondary_y=False, row=2, col=1)

            # ==================================================================================================================== #
            # BOLLINGER BANDS

            bb_upper = go.Scatter(x=ohlc_df.dtime,
                                  y=ohlc_df.Bollinger_Band_High,
                                  mode='lines',
                                  name='BB_high',
                                  visible='legendonly'
                                  )

            bb_lower = go.Scatter(x=ohlc_df.dtime,
                                  y=ohlc_df.Bollinger_Band_Low,
                                  mode='lines',
                                  name='BB_low',
                                  visible='legendonly'
                                  )

            fig.add_trace(bb_upper, secondary_y=False, row=2, col=1)
            fig.add_trace(bb_lower, secondary_y=False, row=2, col=1)


            # ==================================================================================================================== #
            # CHANDELIER EXIT

            CE_upper = go.Scatter(x=ohlc_df.dtime,
                                  y=ohlc_df.CE_upper,
                                  mode='lines',
                                  name='CE_upper',
                                  visible='legendonly'
                                  )

            CE_lower = go.Scatter(x=ohlc_df.dtime,
                                  y=ohlc_df.CE_lower,
                                  mode='lines',
                                  name='CE_lower',
                                  visible='legendonly'
                                  )

            fig.add_trace(CE_upper, secondary_y=False, row=2, col=1)
            fig.add_trace(CE_lower, secondary_y=False, row=2, col=1)

            # ==================================================================================================================== #
            # TRACKLINE

            trackline_upper = go.Scatter(x=ohlc_df.dtime,
                                         y=ohlc_df.trackline_upper,
                                         mode='lines',
                                         name='trackline_upper',
                                         visible='legendonly'
                                         )

            trackline_lower = go.Scatter(x=ohlc_df.dtime,
                                         y=ohlc_df.trackline_lower,
                                         mode='lines',
                                         name='trackline_lower',
                                         visible='legendonly'
                                         )


            trackline_colors = np.where(ohlc_df.trackline_trend == 'up', 'chartreuse',
                                        np.where(ohlc_df.trackline_trend == 'down', 'red',
                                                 np.where(ohlc_df.trackline_trend == 'neutral', 'yellow', 'cyan')))


            ohlc_df['trackline_colors'] = trackline_colors

            print(ohlc_df.trackline_colors)

            trackline = go.Scatter(x=ohlc_df.dtime,
                                   y=ohlc_df.trackline,
                                   mode='markers+lines',
                                   name='trackline',
                                   visible='legendonly',
                                   marker=dict(color=ohlc_df.trackline_colors),
                                   line=dict(color='lightblue')
                                   )

            # trackline = px.line(x=ohlc_df.dtime,
            #                     y=ohlc_df.trackline,
            #                     # color= ohlc_df.trackline_colors
            #                     # line=dict(color=ohlc_df.trackline_colors)
            #                     ).data[0]

            # trackline = px.line(data_frame= ohlc_df,
            #                     x='dtime',
            #                     y='trackline',
            #                     color= 'trackline_colors'
            #                     # line=dict(color=ohlc_df.trackline_colors)
            #                     ).data[0]

            fig.add_trace(trackline_upper, secondary_y=False, row=2, col=1)
            fig.add_trace(trackline_lower, secondary_y=False, row=2, col=1)
            fig.add_trace(trackline, secondary_y=False, row=2, col=1)

            # ==================================================================================================================== #
            # HEIKIN ASHI
            #
            # heiken_ashi = go.Candlestick(
            #     open=ohlc_df.HA_open,
            #     high=ohlc_df.HA_high,
            #     low=ohlc_df.HA_low,
            #     close=ohlc_df.HA_close,
            #     x=ohlc_df.dtime,
            #     name='Heiken Ashi',
            #     visible= 'legendonly'
            # )
            #
            # fig.add_trace(heiken_ashi, row=2, col=1)
            #
            # HA_bb_upper = go.Scatter(x=ohlc_df.dtime,
            #                          y=ohlc_df.HA_Bollinger_Band_High,
            #                          mode='lines',
            #                          name='HA_BB_high',
            #                          visible='legendonly'
            #                          )
            #
            # HA_bb_lower = go.Scatter(x=ohlc_df.dtime,
            #                          y=ohlc_df.HA_Bollinger_Band_Low,
            #                          mode='lines',
            #                          name='HA_BB_low',
            #                          visible='legendonly'
            #                          )
            #
            # HA_EMA_10_trace = go.Scatter(x=ohlc_df.dtime,
            #                              y=ohlc_df.HA_EMA_10,
            #                              mode='lines',
            #                              name='HA_EMA_10',
            #                              visible='legendonly'
            #                              )
            #
            # HA_trackline = go.Scatter(x=ohlc_df.dtime,
            #                           y=ohlc_df.HA_trackline,
            #                           mode='markers+lines',
            #                           name='HA_trackline',
            #                           visible='legendonly',
            #                           marker=dict(color=ohlc_df.trackline_colors),
            #                           line=dict(color='lightblue')
            #                           )
            #
            # fig.add_trace(HA_bb_upper, secondary_y=False, row=2, col=1)
            # fig.add_trace(HA_bb_lower, secondary_y=False, row=2, col=1)
            # fig.add_trace(HA_trackline, secondary_y=False, row=2, col=1)
            #
            # fig.add_trace(HA_EMA_10_trace, secondary_y=False, row=2, col=1)
            # fig.add_trace(go.Scatter(x=ohlc_df.dtime,
            #                          y=ohlc_df.HA_ema_10_derivative,
            #                          mode='lines',
            #                          name='EMA_12_d',
            #                          visible= 'legendonly'
            #                          ), row=3, col=1)
            #

            # ==================================================================================================================== #

            # Update yaxis properties

            fig.update_xaxes(showspikes=True,
                             spikemode='across+toaxis',
                             spikesnap='cursor',
                             spikedash='solid',
                             spikethickness=1,
                             spikecolor='white',
                             showline=True,
                             showgrid=True,
                             row=1, col=1)

            fig.update_xaxes(showspikes=True,
                             spikemode='across+toaxis',
                             spikesnap='cursor',
                             spikedash='solid',
                             spikethickness=1,
                             showline=True,
                             showgrid=True,
                             row=2, col=1)

            fig.update_xaxes(showspikes=True,
                             spikemode='across',
                             spikesnap='cursor',
                             spikedash='solid',
                             spikethickness=1,
                             showline=True,
                             showgrid=True,
                             row=3, col=1)

            # Update yaxis properties
            fig.update_yaxes(title_text="Portfolio Value [$]", row=1, col=1)
            fig.update_yaxes(title_text="USD [$]", row=2, col=1)

            fig.update_layout(
                template='plotly_dark',
                margin=dict(t=30, b=30),
                hoverdistance=1000,
                hovermode='x',
            )

            fig.update_traces(xaxis='x1')

            # ==================================================================================================================== #
            # AUTO SIZE CHARTS

            if relayoutdata is None:

                print('relayoutdata is None')
                pass

            elif 'autosize' in relayoutdata.keys():

                print('relayoutdata has autosize')
                pass

            elif "xaxis.range[0]" in relayoutdata.keys():

                print('xaxis.range in relayoutData')

                try:

                    # Update Y axis scaling using callback as per linked method
                    # https://stackoverflow.com/questions/71029800/how-to-dynamically-change-the-scale-ticks-of-y-axis-in-plotly-charts-upon-zoomin

                    # Filter data to selected timeframe as per boolean mask technique in link below
                    # https://stackoverflow.com/questions/29370057/select-dataframe-rows-between-two-dates

                    first_time = relayoutdata["xaxis.range[0]"]
                    last_time = relayoutdata["xaxis.range[1]"]

                    mask1 = (portfolio_val_df.dtime > first_time) & (portfolio_val_df.dtime <= last_time)
                    d1 = portfolio_val_df.loc[mask1, 'portfolio_val']
                    y1_min = d1.min() * 0.995
                    y1_max = d1.max() * 1.005

                    mask2 = (ohlc_df.dtime > first_time) & (ohlc_df.dtime <= last_time)
                    d2 = ohlc_df.loc[mask2, ['open', 'high', 'low', 'close']]
                    y2_min = d2.min().min() * 0.9
                    y2_max = d2.max().max() * 1.1

                    mask3 = (ohlc_df.dtime > first_time) & (ohlc_df.dtime <= last_time)
                    d3 = ohlc_df.loc[mask3, ['ema_12_derivative']]
                    y3_min = d3.min().min() * 0.995
                    y3_max = d3.max().max() * 1.005

                    if len(d2) > 0:
                        fig["layout"]["yaxis"]["range"] = [y1_min, y1_max]
                        fig["layout"]["yaxis2"]["range"] = [y2_min, y2_max]
                        fig["layout"]["yaxis3"]["range"] = [y3_min, y3_max]

                except KeyError:
                    pass
                finally:
                    x_range = [relayoutdata["xaxis.range[0]"], relayoutdata["xaxis.range[1]"]]
                    fig["layout"]["xaxis"]["range"] = x_range

            else:

                print('relayoutdata IS NONE')

                first_date = ohlc_df.dtime[0]
                last_date = ohlc_df.dtime[ohlc_df.shape[0] - 1]
                x_range = [first_date, last_date]

                fig["layout"]["xaxis"]["range"] = x_range

            return fig

    def create_portfolio_summary_layout(self):

        portfolio_summary_layout = dbc.Container(
            [
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            html.Div(
                                id='initial-time',
                                children=['Initial Time: ' + self._initial_time_data + ' UTC']
                            ),
                            html.Hr()
                        ], style= {'height': '30%'}),

                        dbc.Row([
                            html.Div(
                                id='current-time',
                                children=[]
                            ),
                            html.Hr(),
                            dcc.Interval(
                                id='update-current-time',
                                interval=self._n_intervals_seconds * 1000,  # in milliseconds
                                n_intervals=0,
                                disabled=False
                            )
                        ], style= {'height': '30%'})

                    ], width={'size': 4, 'offset': 0, 'order': 1}),

                    dbc.Col([
                        html.H2('PORTFOLIO DASHBOARD', className='text-center text-primary, mb-3')
                    ], width={'size': 4, 'offset': 0, 'order': 2}),

                ]),


                dbc.Row(
                    [  # start of second row ]
                        dbc.Col([  # first col of second row
                            html.H5('Portfolio Indicators', className='text-center'),
                            dcc.Graph(id='portfolio-indicators',
                                      style={'height': 90},
                                      ),
                            html.Hr(),
                            dcc.Interval(
                                id='update-portfolio-indicators',
                                interval= self._n_intervals_seconds * 1000,  # in milliseconds
                                n_intervals=0,
                                disabled=False
                            )
                        ], width={'size': 12, 'offset': 0, 'order': 1}),
                    ]
                ),

                dbc.Row(
                    [  # start of third row

                        dbc.Col([   # first col of third row
                            html.H5('Total Portfolio Value ($USD)', className='text-center'),
                            dcc.Graph(
                                id='portfolio-value-chart',
                                style={'height': 400}
                            ),
                            html.Hr(),
                            dcc.Interval(
                                id='update-portfolio-value-chart',
                                interval= self._n_intervals_seconds *1000, # in milliseconds
                                n_intervals=0,
                                disabled= False
                            ),
                        ], width = {'size': 8, 'offset': 0, 'order': 1}),

                        dbc.Col([   # second col of third row
                            html.H5('Win Rate', className='text-center'),
                            dcc.Graph(
                                id='win-rate-chart',
                                style={'height': 400}
                            ),
                            html.Hr(),
                            dcc.Interval(
                                id='update-win-rate-chart',
                                interval= self._n_intervals_seconds * 1000,  # in milliseconds
                                n_intervals=0,
                                disabled=False
                            )
                        ], width = {'size': 4, 'offset': 0, 'order': 2}),

                    ], justify= 'end', className= 'g-0',
                ),

                dbc.Row([  # start of fourth row ]

                    dbc.Col([   # first col of fourth row
                        html.H5('Open Positions', className='text-center'),
                        dcc.Graph(id='open-positions-table',
                                  style={'height': 300}),
                        dcc.Interval(
                            id='update-open-positions-table',
                            interval= self._n_intervals_seconds * 1000, # in milliseconds
                            n_intervals=0,
                            disabled= False
                        )
                    ], width={'size': 12, 'offset': 0, 'order': 2}),
                ])
            ], fluid= True
        )

        return portfolio_summary_layout

    def create_candlestick_charts_layout(self):

        candlestick_chart_layout = dbc.Container(

            [

                dbc.Row([  # start of second row

                    dbc.Col([  # first col of second row
                        html.H5('SELECT PAIR', className='text-center'),
                        dcc.Dropdown(self._pairs, self._pairs[0], clearable= False, id='pair')

                    ], width={'size': 2, 'offset': 0, 'order': 1}),

                    dbc.Col([  # first col of third row
                        html.H5('SELECT TIMEFRAME', className='text-center'),
                        dcc.Dropdown(self._timeframes, self._timeframes[0], clearable= False, id='timeframes')

                    ], width={'size': 2, 'offset': 0, 'order': 2}),

                    dbc.Col(html.H2('CANDLESTICK CHARTS',
                                    className='text-center text-primary, mb-3'
                                    ), width={'size': 4, 'offset': 0, 'order': 3})  # header row

                ]),

                dbc.Row([  # start of third row ]

                    dbc.Col([  # first col  of third row
                        dcc.Graph(id='candlestick_chart',
                                  figure={},
                                  style={'height': 1000})

                    ], width={'size': 12, 'offset': 0, 'order': 3})

                ])

            ], fluid=True

        )

        return candlestick_chart_layout

    def start(self):
        print('Starting Portfolio Dashboard')
        self._app.run_server(debug= False, host= '0.0.0.0', port= int(os.environ.get("PORT", 5000)))
        webbrowser.open_new('http://127.0.0.1:5000/')

    def update_portfolio_visuals_params(self, initial_time, initial_cash, pairs, timeframes,
                                        current_time, portfolio_value_time_list, portfolio_value_list,
                                        open_position_table_df, trade_win_rate_df, trade_win_rate_rolling_df, total_num_trades,
                                        trade_return_percent, unrealized_PnL):

        self._initial_time = initial_time
        self._initial_equity = initial_cash
        self._pairs = pairs
        self._timeframes = timeframes

        self._current_time = current_time
        self._time_list = portfolio_value_time_list
        self._portfolio_value_list = portfolio_value_list
        self._open_position_table_df = open_position_table_df

        self._trade_win_rate_df = trade_win_rate_df
        self._trade_win_rate_rolling_df = trade_win_rate_rolling_df
        self._total_num_trades = total_num_trades
        self._trade_return_percent = trade_return_percent
        self._unrealized_PnL = unrealized_PnL

class main_to_dash_bridge:

    _open_positions_table_header = ['pair', 'type', 'open_volume', 'entry_price', 'current_price',
                                    'leverage', 'pnl']

    def __init__(self, trade_performance_monitoring_path, current_time_csv_filename,
                 initial_data_csv_filename,
                 portfolio_value_time_list_csv_filename,
                 portfolio_value_list_csv_filename,
                 open_position_table_df_filename,
                 trade_win_rate_df_csv_filename,
                 trade_win_rate_rolling_df_csv_filename,
                 total_num_trades_csv_filename,
                 trade_return_percent_csv_filename,
                 unrealized_PnL_csv_filename,
                 initial_data_df = None):

        self.trade_performance_monitoring_path = trade_performance_monitoring_path
        self.initial_data_df = initial_data_df

        self.current_time_csv_file_path = utils.create_csv_file_path(self.trade_performance_monitoring_path, current_time_csv_filename)
        self.initial_data_csv_file_path = utils.create_csv_file_path(self.trade_performance_monitoring_path, initial_data_csv_filename)
        self.portfolio_value_time_list_csv_file_path = utils.create_csv_file_path(self.trade_performance_monitoring_path, portfolio_value_time_list_csv_filename)
        self.portfolio_value_list_csv_file_path = utils.create_csv_file_path(self.trade_performance_monitoring_path, portfolio_value_list_csv_filename)
        self.open_position_table_df_file_path = utils.create_csv_file_path(self.trade_performance_monitoring_path, open_position_table_df_filename)
        self.trade_win_rate_df_csv_file_path = utils.create_csv_file_path(self.trade_performance_monitoring_path, trade_win_rate_df_csv_filename)
        self.trade_win_rate_rolling_df_csv_file_path = utils.create_csv_file_path(self.trade_performance_monitoring_path, trade_win_rate_rolling_df_csv_filename)
        self.total_num_trades_csv_file_path = utils.create_csv_file_path(self.trade_performance_monitoring_path, total_num_trades_csv_filename)
        self.trade_return_percent_csv_file_path = utils.create_csv_file_path(self.trade_performance_monitoring_path, trade_return_percent_csv_filename)
        self.unrealized_PnL_csv_file_path = utils.create_csv_file_path(self.trade_performance_monitoring_path, unrealized_PnL_csv_filename)

    def write_initial_data_df(self):

        self.initial_data_df.to_csv(self.initial_data_csv_file_path)

    def read_initial_data_from_csv(self):

        initial_data_df = utils.read_csv_data(csv_file_path= self.initial_data_csv_file_path)



        try:

            initial_time = initial_data_df.at[0, 'initial_time']
            initial_cash = initial_data_df.at[0, 'initial_cash']
            pairs = ast.literal_eval(initial_data_df.at[0, 'pairs'])
            timeframes = ast.literal_eval(initial_data_df.at[0, 'timeframes'])

        except:

            initial_time = 0
            initial_cash = 0
            pairs = []
            timeframes = []

        return initial_time, initial_cash, pairs, timeframes


    def write_visual_data_to_csv(self, current_time, portfolio_value_time_list, portfolio_value_list, open_position_table_df,
                                 trade_win_rate_df, trade_win_rate_rolling_df, total_num_trades, trade_return_percent,
                                 unrealized_PnL):

        current_time_df = pd.DataFrame({'col1': [current_time]})
        portfolio_value_time_list_df = pd.DataFrame({'col1': portfolio_value_time_list})
        portfolio_value_list_df = pd.DataFrame({'col1': portfolio_value_list})
        total_num_trades_df = pd.DataFrame({'col1': [total_num_trades]})
        trade_return_percent_df = pd.DataFrame({'col1': [trade_return_percent]})
        unrealized_PnL_df = pd.DataFrame({'col1': [unrealized_PnL]})

        current_time_df.to_csv(self.current_time_csv_file_path)
        portfolio_value_time_list_df.to_csv(self.portfolio_value_time_list_csv_file_path)
        portfolio_value_list_df.to_csv(self.portfolio_value_list_csv_file_path)
        open_position_table_df.to_csv(self.open_position_table_df_file_path)
        trade_win_rate_df.to_csv(self.trade_win_rate_df_csv_file_path)
        trade_win_rate_rolling_df.to_csv(self.trade_win_rate_rolling_df_csv_file_path)
        total_num_trades_df.to_csv(self.total_num_trades_csv_file_path)
        trade_return_percent_df.to_csv(self.trade_return_percent_csv_file_path)
        unrealized_PnL_df.to_csv(self.unrealized_PnL_csv_file_path)


    def read_visual_data_from_csv(self):

        initial_time, initial_cash, pairs, timeframes = self.read_initial_data_from_csv()

        current_time = utils.read_csv_data(csv_file_path= self.current_time_csv_file_path)
        time_list = utils.read_csv_data(csv_file_path= self.portfolio_value_time_list_csv_file_path)
        portfolio_value_list = utils.read_csv_data(csv_file_path= self.portfolio_value_list_csv_file_path)
        open_positions_table_df = utils.read_csv_data(csv_file_path= self.open_position_table_df_file_path)
        trade_win_rate_df = utils.read_csv_data(csv_file_path= self.trade_win_rate_df_csv_file_path)
        trade_win_rate_rolling_df = utils.read_csv_data(csv_file_path= self.trade_win_rate_rolling_df_csv_file_path)
        total_num_trades = utils.read_csv_data(csv_file_path= self.total_num_trades_csv_file_path)
        trade_return_percent = utils.read_csv_data(csv_file_path= self.trade_return_percent_csv_file_path)
        unrealized_PnL = utils.read_csv_data(csv_file_path= self.unrealized_PnL_csv_file_path)

        try:
            current_time = current_time.at[0, 'col1']
        except:
            current_time = str(0)

        try:
            time_list = time_list.col1.to_list()
        except:
            time_list = ['1970-01-01']

        try:
            portfolio_value_list = portfolio_value_list.col1.to_list()
        except:
            portfolio_value_list = [0]

        try:
            is_position_table = open_positions_table_df.at[0, 'pair']
        except:
            open_positions_table_df = pd.DataFrame(columns= self._open_positions_table_header)

        try:
            is_trade_win_rate_df = trade_win_rate_df.at[0, 'win_rate']
        except:
            trade_win_rate_df = pd.DataFrame({'time': [0], 'win_rate': [0]})

        try:
            is_trade_win_rate_rolling_df = trade_win_rate_rolling_df.at[0, 'rolling_win_rate']
        except:
            trade_win_rate_rolling_df = pd.DataFrame({'time': [0], 'rolling_win_rate': [0]})

        try:
            total_num_trades = total_num_trades.at[0, 'col1']
        except:
            total_num_trades = 0

        try:
            trade_return_percent = trade_return_percent.at[0, 'col1']
        except:
            trade_return_percent = 0

        try:
            unrealized_PnL = unrealized_PnL.at[0, 'col1']
        except:
            unrealized_PnL = 0


        return initial_time, initial_cash, pairs, timeframes, \
               current_time, time_list, portfolio_value_list, open_positions_table_df, \
               trade_win_rate_df, trade_win_rate_rolling_df,total_num_trades, \
               trade_return_percent, unrealized_PnL
