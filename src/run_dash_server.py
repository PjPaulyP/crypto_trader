import trade_performance_monitoring as tpm
import time
import os

ohlc_csv_folder_name = '../data/ohlc_data'
trade_performance_monitoring_folder_name = '../data/trade_performance_monitoring'
current_time_csv_filename = 'current_time'
initial_data_csv_filename = 'initial_data'
portfolio_value_time_list_csv_filename = 'portfolio_value_time_list'
portfolio_value_list_csv_filename = 'portfolio_value_list'
open_poition_table_df_filename = 'open_position_table_df'
trade_win_rate_df_csv_filename = 'trade_win_rate_df'
trade_win_rate_rolling_df_csv_filename = 'trade_win_rate_rolling_df'
total_num_trades_csv_filename = 'total_num_trades'
trade_return_percent_csv_filename = 'trade_return_percent'
unrealized_PnL_csv_filename = 'unrealized_PnL'


MTD_BRDGE_DASH = tpm.main_to_dash_bridge(trade_performance_monitoring_folder_name,
                                         current_time_csv_filename, initial_data_csv_filename,
                                         portfolio_value_time_list_csv_filename, portfolio_value_list_csv_filename,
                                         open_poition_table_df_filename, trade_win_rate_df_csv_filename,
                                         trade_win_rate_rolling_df_csv_filename, total_num_trades_csv_filename,
                                         trade_return_percent_csv_filename, unrealized_PnL_csv_filename)

is_initial_data = True
initial_data_filepath = trade_performance_monitoring_folder_name + '/' + initial_data_csv_filename + '.csv'

while is_initial_data == True:

    if os.path.isfile(initial_data_filepath) or os.path.islink(initial_data_filepath):
        is_initial_data = False
    else:
        print('Waiting for initial data to populate from main function')
        time.sleep(3)

initial_time, initial_cash, pairs, timeframes = MTD_BRDGE_DASH.read_initial_data_from_csv()

PV = tpm.portfolio_visuals(initial_time, initial_cash, pairs, timeframes,
                           ohlc_csv_folder_name, trade_performance_monitoring_folder_name, portfolio_value_list_csv_filename, portfolio_value_time_list_csv_filename,
                           main_to_dash_bridge_obj= MTD_BRDGE_DASH)

if __name__ == "__main__":

    PV.start()
