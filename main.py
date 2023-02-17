import time
import pandas as pd
import kraken_excel_interface as kei
import trade_model as tm
import utility as utils
import kraken_exchange_tester as ket
from datetime import datetime
import trade_performance_monitoring as tpm
import sys
import numpy as np


# pairs = ['ETHUSD', 'XBTUSD', 'MATICUSD', 'SOLUSD', 'ADAUSD']
pairs = ['MATICUSD', 'SOLUSD', 'ADAUSD']
# pairs = ['MATICUSD']
timeframes = [60, 240, 360, 1440]  # Candle stick intervals in minutes
initial_cash = 1000.00  # For testing.
max_trade_val_percent = 0.10
leverage = 1  # Leverage (int). I.e., '2' = '2:1' leverage, '1' = '1:1' leverage, etc...
is_test_mode = True

# Initiate trade models.
# STM is the sub trade model which the central trade model uses.
# STM = tm.stoch_rsi_model()
EMA_cross_model = tm.EMA_cross_model()
STM = tm.EMA_cross_model_cross_timeframe_voting(EMA_cross_model)
# STM = tm.support_resistance_model()
# STM = tm.bollinger_derivative_reversal_model(pairs= pairs)
# STM = tm.bollinger_derivative_reversal_model_heikin_ashi(pairs= pairs)
# STM = tm.trackline_follower_model()
# STM = tm.trackline_follower_model(is_heikin_ashi= True)

class main:

    ### Intialize variables
    initial_time = datetime.timestamp(datetime.now())
    current_time = initial_time
    end_time = None

    ohlc_csv_folder_name = '../data/ohlc_data'
    trade_csv_folder_name = '../data/trade_data'
    keys_text_name = '../resources/kraken_keys/keys.txt'
    test_data_path = '../data/past_crypto_test_data'

    trade_performance_monitoring_folder_name = '../data/trade_performance_monitoring'
    current_time_csv_filename = 'current_time'
    initial_data_csv_filename = 'initial_data'
    portfolio_value_time_list_csv_filename = 'portfolio_value_time_list'
    portfolio_value_list_csv_filename = 'portfolio_value_list'
    open_position_table_df_filename = 'open_position_table_df'
    account_balance_filename = 'account_balance'
    open_spot_positions_filename = 'open_spot_positions'
    trade_win_rate_df_csv_filename = 'trade_win_rate_df'
    trade_win_rate_rolling_df_csv_filename = 'trade_win_rate_rolling_df'
    total_num_trades_csv_filename = 'total_num_trades'
    trade_return_percent_csv_filename = 'trade_return_percent'
    unrealized_PnL_csv_filename = 'unrealized_PnL'
    terminal_output_filename = 'terminal_output'

    last_unixtime_df = pd.DataFrame(index=pairs, columns=timeframes)
    for col in last_unixtime_df.columns:
        last_unixtime_df[col].values[:] = None
    tradesdf = pd.DataFrame({'pair': pairs})

    ### Delete existing data excel sheet
    utils.delete_old_files(ohlc_csv_folder_name)
    utils.delete_old_files(trade_csv_folder_name)
    utils.delete_old_files(trade_performance_monitoring_folder_name)

    # Create file to log terminal output for troubleshooting
    terminal_output_filepath = trade_performance_monitoring_folder_name + '/' + terminal_output_filename + '.txt'
    np.set_printoptions(threshold= np.inf)
    sys.stdout = open(terminal_output_filepath, 'w')

    ### INITIALIZE KRAKEN EXCEL INTERFACE
    if is_test_mode == False:

        EE = None

    elif is_test_mode == True:
        EE = ket.kraken_exchanger_tester(ohlc_csv_test_data_folder_path=test_data_path,
                                         trade_csv_folder_path=trade_csv_folder_name,
                                         pairs_list=pairs, initial_cash_balance=initial_cash,
                                         timeframes_list= timeframes)


    KEI = kei.KrakenExcelInterface(initial_time, keys_text=keys_text_name, pairs_list= pairs, timeframe_list= timeframes,
                                   ohlc_csv_folder_name=ohlc_csv_folder_name, trade_csv_folder_name=trade_csv_folder_name,
                                   open_spot_positions_filename= open_spot_positions_filename, test_data_path=test_data_path,
                                   kraken_exchange_tester= EE)

    # Intialize min volume size for each trading pair
    # Initialize price decimal places
    pair_min_trade_volume_dict, pair_max_trade_price_decimal = KEI.update_trade_asset_info(pairs= pairs)

    ### INITIALIZE OHLC + TRADE EXCEL SHEETS. Update last_unixtime_df
    last_unixtime_df = KEI.init_ohlc_data_all(trades_df=tradesdf, last_unixtime_df=last_unixtime_df, timeframe=timeframes)[1]


    # INITIALIZE IF NOT IN TEST MODE
    if is_test_mode == False:

        pair_price_dict = KEI.get_pair_price()
        all_trade_df, all_trade_csv_file_names, account_balance_df = KEI.get_Kraken_acc_data()
        KEI.append_trade_excel_data(all_trade_df, all_trade_csv_file_names, is_first_data=True)
        initial_account_trade_balance = KEI.update_trade_account_balance()
        initial_cash = float(KEI.get_Kraken_acc_data()[2].at['ZUSD', 'vol'])

    else:
        EE.increment_test_time(updated_current_unixtime= EE.first_iteration_end_time) # Update test time to end of initialization data
        initial_time = EE.current_test_time
        current_time = EE.current_test_time
        pair_price_dict = KEI.get_pair_price()
        EE.calc_curr_free_margin()
        EE.get_trade_balance()
        all_trade_df, trade_csv_file_names = EE.all_trade_df, EE.trade_csv_file_names
        KEI.append_trade_excel_data(all_trade_df, trade_csv_file_names, is_first_data=True)
        initial_account_trade_balance = KEI.update_trade_account_balance()

        print('1.0 time t/s ---------------')
        print(last_unixtime_df)
        print(EE.current_test_time)

    TM = tm.trade_model_to_main_interface(initial_cash=initial_cash, max_trade_val_percent=max_trade_val_percent,
                                          leverage= leverage, trades_df=tradesdf, last_unixtime_df=last_unixtime_df,
                                          timeframe_list= timeframes, pair_price_dict= pair_price_dict,
                                          pair_min_trade_volume_dict= pair_min_trade_volume_dict, pair_max_trade_price_decimal= pair_max_trade_price_decimal,
                                          ohlc_csv_folder_name=ohlc_csv_folder_name, trade_csv_folder_name=trade_csv_folder_name, trade_model= STM)

    TPM = tpm.trade_performance_monitoring(trade_csv_folder_path= trade_csv_folder_name,
                                           trade_perf_monitoring_folder_path= trade_performance_monitoring_folder_name,
                                           initial_time= initial_time,
                                           account_trade_balance_df= initial_account_trade_balance,
                                           pair_price_dict= pair_price_dict,
                                           pairs_list= pairs)



    initial_visual_data_df = pd.DataFrame({'initial_time': [initial_time], 'initial_cash': [initial_cash], 'pairs': [pairs], 'timeframes': [timeframes]})
    MTD_BRDGE = tpm.main_to_dash_bridge(trade_performance_monitoring_folder_name,
                                        current_time_csv_filename, initial_data_csv_filename,
                                        portfolio_value_time_list_csv_filename, portfolio_value_list_csv_filename,
                                        open_position_table_df_filename, trade_win_rate_df_csv_filename,
                                        trade_win_rate_rolling_df_csv_filename, total_num_trades_csv_filename,
                                        trade_return_percent_csv_filename, unrealized_PnL_csv_filename,
                                        initial_visual_data_df)
    MTD_BRDGE.write_initial_data_df()

    def __init__(self, pairs: list, timeframes: list, initial_cash: float, max_trade_val_percent: float, max_trade_loss_percent: float,
                 leverage: int, is_test_mode: bool):

        self.pairs = pairs
        self.timeframes = timeframes   # Candle stick intervals in minutes
        self.initial_cash = initial_cash  # Float. If test mode == False, this gets overwritten by initial account cash balance below
        self.max_trade_val_percent = max_trade_val_percent
        self.max_trade_loss_percent = max_trade_loss_percent  # Ex. 0.02 = 2%
        self.leverage = leverage  # Leverage (int). I.e., '2' = '2:1' leverage, '1' = '1:1' leverage, etc...
        self.is_test_mode = is_test_mode

    def main(self):

        if self.is_test_mode == False:
            self._main_loop()
        else:
            self._main_loop_test()

    def _main_loop(self):

        while True:

            # Update variables at start of the loop.
            self.current_time = datetime.timestamp(datetime.now()) # Current test time
            cash = self.KEI.get_Kraken_acc_data()[2].at['ZUSD', 'vol'] # Account cash
            self.TM.update_max_trade_val(cash) # Update max trade val based on current cash
            pair_min_trade_volume_dict, pair_max_trade_price_decimal = self.KEI.update_trade_asset_info(pairs=pairs) # Update pair min vol and decimal
            self.TM.pair_min_trade_volume_dict = pair_min_trade_volume_dict
            self.TM.pair_max_trade_price_decimal = pair_max_trade_price_decimal

            print('=============================================================================================================================================')
            print(datetime.utcfromtimestamp(self.current_time).strftime('%Y-%m-%d %H:%M:%S'))
            print(self.current_time)

            self.last_unixtime_df = self.KEI.append_ohlc_data_all(self.tradesdf, self.last_unixtime_df, self.timeframes)[1]

            # Pair price dict comes after ohlc data gets appended to get same pricing data @ current time
            pair_price_dict = self.KEI.get_pair_price(timeframe= self.timeframes[0]) # Any timeframe is fine...just need to grab last data point on any OHLC
            self.TM.pair_price_dict = pair_price_dict
            print('Pair Price Dict')
            print(pair_price_dict)

            # Update trade data, incase an open order was triggered
            all_trade_df, all_trade_csv_file_names, account_balance_df = self.KEI.get_Kraken_acc_data()
            self.KEI.append_trade_excel_data(all_trade_df, all_trade_csv_file_names, is_first_data=False)
            self.KEI.write_account_balance_data(account_balance_df= account_balance_df,
                                                account_balance_filename= self.account_balance_filename)

            # Return new orders, close orders and cancel orders from trade model
            # Execute these orders
            new_orders_df, close_orders_df, cancel_orders_df, updated_stop_orders_df, cancel_trail_stop_orders_df = self.TM.is_trade()

            print('5.0 NEW ORDERS ------------------------------------')
            print(new_orders_df, close_orders_df, cancel_orders_df)

            self.KEI.loop_kraken_interface_functions(new_order_info_df=new_orders_df,
                                                     close_order_info_df=close_orders_df,
                                                     cancel_orders_df= cancel_orders_df,
                                                     updated_stop_orders_df= updated_stop_orders_df,
                                                     cancel_traiL_stop_orders_df= cancel_trail_stop_orders_df)

            # Update orders csv's
            all_trade_df, all_trade_csv_file_names, account_balance_df = self.KEI.get_Kraken_acc_data()
            self.KEI.append_trade_excel_data(all_trade_df, all_trade_csv_file_names, is_first_data=False)
            self.KEI.write_account_balance_data(account_balance_df= account_balance_df,
                                                account_balance_filename= self.account_balance_filename)

            # Update and return trade account balance, then update trade performance monitoring object
            account_trade_balance = self.KEI.update_trade_account_balance()
            open_spot_positions = utils.read_csv_data(csv_folder_name=self.trade_csv_folder_name,
                                                      csv_file_name=self.open_spot_positions_filename)

            self.TPM.loop_trade_perf_functions(account_trade_balance, open_spot_positions, self.current_time, pair_price_dict)

            print(account_trade_balance)
            print(account_balance_df)

            # Write data to trade_performance_monitoring folder for dashboard app
            self.MTD_BRDGE.write_visual_data_to_csv(self.current_time, self.TPM.portfolio_value_df.time.to_list(),
                                                    self.TPM.portfolio_value_df.portfolio_value.to_list(),
                                                    self.TPM.open_positions_table_df, self.TPM.trade_win_rate_df,
                                                    self.TPM.trade_win_rate_rolling_df, self.TPM.total_num_trades,
                                                    self.TPM.trade_return_percent, self.TPM.unrealized_PnL)

            time.sleep(20)

    def _main_loop_test(self):

        while True:

            # Update variables at start of the loop.
            self.current_time = self.EE.current_test_time # Current test time
            cash = self.KEI.get_Kraken_acc_data()[2].at['ZUSD', 'vol'] # Account cash
            self.TM.update_max_trade_val(cash) # Update max trade val based on current cash

            print('=============================================================================================================================================')
            print(datetime.utcfromtimestamp(self.current_time).strftime('%Y-%m-%d %H:%M:%S'))
            print(self.current_time)

            self.last_unixtime_df = self.KEI.append_ohlc_data_all(self.tradesdf, self.last_unixtime_df, self.timeframes)[1]

            # Pair price dict comes after ohlc data gets appended to get same pricing data @ current time
            pair_price_dict = self.KEI.get_pair_price()
            self.TM.pair_price_dict = pair_price_dict
            print('Pair Price Dict')
            print(pair_price_dict)
            print(self.EE.pairs_price_dict)

            # Execute open orders on test exchange. Do this before the trade model is_trade to avoid orders being generated
            # at the same time an open order being executed. Then update csv data.
            # Also update cash amount
            self.EE.loop_execute_open_order()
            self.EE.calc_curr_free_margin()
            cash = self.KEI.get_Kraken_acc_data()[2].at['ZUSD', 'vol'] # Account cash
            self.TM.update_max_trade_val(cash) # Update max trade val based on current cash

            # Update trade data, incase an open order was triggered
            all_trade_df, all_trade_csv_file_names, account_balance_df = self.KEI.get_Kraken_acc_data()
            self.KEI.append_trade_excel_data(all_trade_df, all_trade_csv_file_names, is_first_data=False)
            self.KEI.write_account_balance_data(account_balance_df= account_balance_df,
                                                account_balance_filename= self.account_balance_filename)

            # Return new orders, close orders and cancel orders from trade model
            # Execute these orders
            new_orders_df, close_orders_df, cancel_orders_df, updated_stop_orders_df, cancel_trail_stop_orders_df = self.TM.is_trade()

            print('\n')
            print('2.0 ORDERS ')
            print('\n')
            print('New Orders DF')
            print(new_orders_df)
            print('\n')
            print('Close  Orders DF')
            print(close_orders_df)
            print('\n')
            print('Cancel Orders DF')
            print(cancel_orders_df)
            print('\n')
            print('Updated Stop Orders DF')
            print(updated_stop_orders_df)
            print('\n')
            print('Cancel Trail Stop Orders DF')
            print(cancel_trail_stop_orders_df)


            self.KEI.loop_kraken_interface_functions(new_order_info_df=new_orders_df,
                                                     close_order_info_df=close_orders_df,
                                                     cancel_orders_df= cancel_orders_df,
                                                     updated_stop_orders_df= updated_stop_orders_df,
                                                     cancel_traiL_stop_orders_df= cancel_trail_stop_orders_df)

            # Update orders csv's
            all_trade_df, all_trade_csv_file_names, account_balance_df = self.KEI.get_Kraken_acc_data()
            self.KEI.append_trade_excel_data(all_trade_df, all_trade_csv_file_names, is_first_data=False)
            self.KEI.write_account_balance_data(account_balance_df= account_balance_df,
                                                account_balance_filename= self.account_balance_filename)

            # Update and return trade account balance, then update trade performance monitoring object
            account_trade_balance = self.KEI.update_trade_account_balance()
            open_spot_positions = utils.read_csv_data(csv_folder_name= self.trade_csv_folder_name, csv_file_name= self.open_spot_positions_filename)

            self.TPM.loop_trade_perf_functions(account_trade_balance, open_spot_positions, self.current_time, pair_price_dict)

            print(account_trade_balance)
            print(account_balance_df)

            # Write data to trade_performance_monitoring folder for dashboard app
            self.MTD_BRDGE.write_visual_data_to_csv(self.current_time, self.TPM.portfolio_value_df.time.to_list(),
                                                    self.TPM.portfolio_value_df.portfolio_value.to_list(),
                                                    self.TPM.open_positions_table_df, self.TPM.trade_win_rate_df,
                                                    self.TPM.trade_win_rate_rolling_df, self.TPM.total_num_trades,
                                                    self.TPM.trade_return_percent, self.TPM.unrealized_PnL)

            self.EE.increment_test_time()

if __name__ == "__main__":

    MAIN = main(pairs= pairs, timeframes= timeframes, initial_cash= initial_cash, max_trade_val_percent= max_trade_val_percent, max_trade_loss_percent= max_trade_val_percent,
                leverage= leverage, is_test_mode= is_test_mode)

    MAIN.main()