import csv
import pandas as pd
import os
from datetime import datetime
import math
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
import time
from functools import wraps
import logging


def delete_old_files(folder_path):

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def write_to_csv(folder_path, file_name, data_frame, replace_last_N_rows= None):

    file_path = folder_path + '/' + file_name + '.csv'

    if os.path.isfile(file_path) or os.path.islink(file_path):
        csv_write_mode = "a"
        is_add_header = False
    else:
        csv_write_mode = "w"
        is_add_header = True

    # Replace last row with updated data point, if is_replace_last_row = True
    # is_replace_last_row is True when unixtime of last row equals last data point of append data
    # Sets lines equal to csv data, excluding last row, then
    if replace_last_N_rows != None:
        with open(file_path) as f:
            lines = f.readlines()[:-replace_last_N_rows]
        with open(file_path, 'w') as f:
            f.writelines(lines)

    # Write/Append to CSV
    data_frame.to_csv(file_path, mode= csv_write_mode, header= is_add_header)

def read_csv_data(csv_folder_name= None, csv_file_name= None, LimitPrevNumRows= None, csv_file_path= None):

    if csv_file_path == None:
        csv_file_path = csv_folder_name + '/' + csv_file_name + '.csv'

    try:

        ### LimitPrevNumRows = number of data points to grab

        with open(csv_file_path) as f:
            csv_file_object = csv.reader(f)
            row_count = sum(1 for row in csv_file_object)

        if LimitPrevNumRows == None: ### NEED TO DO THIS COMPARISON FIRST GIVEN HANDLING OF INT vs. NONE COMPARISON CAN'T BE DONE
            skip_row = None
        elif row_count > LimitPrevNumRows:
            skip_row_to = row_count - LimitPrevNumRows ### Grabs last X data points
            skip_row = range(1, skip_row_to) ### Header = row 0, so skip row=1 to row=skip_row_to
        else:
            skip_row = None

        data_df = pd.read_csv(filepath_or_buffer= csv_file_path, skiprows= skip_row)

        return data_df

    except:

        data_df = pd.DataFrame({'0': [0]})

        return data_df

def reverse_df_order(dataframe: pd.DataFrame):

    ### Kraken displays the Open Orders & Closed Orders from most recent to oldest so adjust our dataframes for that
    reordered_df = dataframe.reindex(index = dataframe.index[::-1])
    return reordered_df

def create_order_tx_id(pair, trade_type):

    if pair == None:
        return 0
    else:
        presentTime = datetime.now()
        unix_timestamp = datetime.timestamp(presentTime) * 1000
        tx_id = pair + '_' + trade_type + '_' + str(unix_timestamp)

    return tx_id

def calculate_position_pnl(trade_type: str, trade_opening_price: float, pair_current_price: float, trade_current_vol: float):

    if trade_type == 'buy':
        position_pnl = (pair_current_price - trade_opening_price) * trade_current_vol
    else:
        position_pnl = (trade_opening_price - pair_current_price ) * trade_current_vol

    return position_pnl

def round_nearest_hundred(value, min_or_max):

    if min_or_max == 'min':

        return int(math.floor(value / 100.0)) * 100

    else:

        return int(math.ceil(value / 100.0)) * 100

def find_first_match_of_two_lists(list1: list, list2:list):

    # Ex - List1 = input times_list
    # List2 = std Kraken API times

    is_list1_in_list2 = [e in list2 for e in list1]

    min_index_true = is_list1_in_list2.index(True)
    first_matching_element = list1[min_index_true]

    return first_matching_element

def apply_rolling_average_list(a_list, n):

    a_list = np.cumsum(a_list, dtype= float)
    a_list[n:] = a_list[n:] - a_list[:-n]
    a_list = a_list[n - 1:] / n

    return a_list

def create_csv_file_path(folder_path, file_name):

    file_path = folder_path + '/' + file_name + '.csv'

    return file_path

def determine_stop_loss_execute(order_info_df: pd.DataFrame, pair_price: float):

    '''

    :param order_info_df:
    :param pair_price:
    :return: (Bool) Returns true if stop loss should execute
    '''

    is_stop_loss_execute = False

    if order_info_df.at[0, 'type'] == 'sell':
        if order_info_df.at[0, 'price'] >= pair_price:
            is_stop_loss_execute = True
    elif order_info_df.at[0, 'type'] == 'buy':
        if order_info_df.at[0, 'price'] <= pair_price:
            is_stop_loss_execute = True

    return is_stop_loss_execute

def next_occurrence(lst: list, value, comparative= None):

    '''
    Note, the returning list will be indexed starting from 0

    :param lst:
    :param current_index:
    :param value:
    :param comparative:
    :return:
    '''

    # Function copied from ChatGPT

    try:
    # Return the index of the value using the index method

        if comparative == None:
            return lst.index(value)

        elif comparative == 'greater-equal':

            bool_lst = np.where(np.array(lst) >= value, True, False)
            ind = list(bool_lst).index(True)

            return ind

        elif comparative == 'lesser-equal':

            bool_lst = np.where(np.array(lst) <= value, True, False)
            ind = list(bool_lst).index(True)
            return ind

    except ValueError:
    # If the value is not found, return -1
        return False

def last_occurence(lst: list, value):

    return len(lst) - lst[-1::-1].index(value) - 1

def convert_unixtime_to_date(unixtime):

    return datetime.utcfromtimestamp(unixtime).strftime('%Y-%m-%d %H:%M:%S')

def convert_pair_to_ticker(pair: str):

    ticker = pair.replace('USD', '')
    ticker = correct_if_x_ticker(ticker)

    return ticker

def convert_ticker_to_pair(ticker: str):

    ticker = correct_if_x_ticker(ticker)
    pair = ticker + 'USD'

    return pair

def correct_if_x_ticker(ticker: str):

    # Some exceptions given ETH and XBT ticker can have an X in front.
    # X tickers are for spot versions of the asset
    ticker = 'ETH' if ticker == 'XETH' else ticker
    ticker = 'XBT' if ticker == 'XXBT' else ticker

    return ticker

def calc_derivative_array(y: pd.Series, is_include_initial_val: bool= False):

    '''
    For input pd.Series, returns corresponding derivative array (np.ndarray). The "X" component must be unit vector (i.e equispaced by units of "1"

    :param ohlc_df:
    :return: np.ndarray
    '''

    dx = 1

    dy = np.diff(y) / dx

    if is_include_initial_val == True:
        dy = np.append([[0]], dy)

    return dy

def k_means_get_optimum_clusters(df: pd.DataFrame, num_data_points: int= 60):

    '''
        Credits/methodlogy/resource for this function linked below.

        https://github.com/judopro/Stock_Support_Resistance_ML/blob/master/find_support_resistance_kmeans.py

    Use K-means clustering to find center of K number cluster of data points.

    :param df:
    :param num_data_points:
    :return:
    '''


    inertia_list = []
    k_models = []

    size = min(num_data_points, df.shape[0])
    for i in range(1, size):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        inertia_list.append(kmeans.inertia_)
        k_models.append(kmeans)

    x = range(1, len(inertia_list) + 1)
    optimum_k = KneeLocator(x, inertia_list, curve='convex', direction='decreasing').knee

    print(optimum_k)
    print(type(optimum_k))

    print("Optimum K is " + str(optimum_k))
    optimum_clusters = k_models[optimum_k - 1]

    return optimum_clusters



def calc_rsi(ohlc, N: int, is_initial_data: bool):

    '''
    Calculate RSI for a pandas series of closing price data, as per Investopedia definition.
    (https://www.investopedia.com/terms/r/rsi.asp)

    Note that the TA library (Version 0.10.1, Bukosabino) uses a non-standard RSI calculation. The weighting of
    N data points uses an EWM based smoothing function vs. the standard one defined on Investopedia

    Spreadsheet Example of RSI calculation linked here:
    (https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi)

    :param close: (pd.Series) Pandas series of closing price data
    :param N: (int) N data points to use for RSI calculation
    :return: (pd.Series) Returns a pandas series of RSI values
    '''

    close = ohlc.close

    diff = close.diff(1)
    up_dir = diff.where(diff > 0, 0.0)
    down_dir = -diff.where(diff < 0, 0.0)

    # RSI Step 1
    # Exclude first value (index = 0) because not valid diff value for avg calc
    avg_initial = pd.Series({close.index[0]: np.NaN})

    if is_initial_data == True:
        # Go up until index val N+1, to apply step 1 avg calc
        avg_up = up_dir[1:N + 1].rolling(window=N, min_periods=N).mean()
        avg_up = pd.concat([avg_initial, avg_up])

        avg_down = down_dir[1:N + 1].rolling(window=N, min_periods=N).mean()
        avg_down = pd.concat([avg_initial, avg_down])

    else:
        avg_up = ohlc.RS_up
        avg_down = ohlc.RS_down


    # RSI Step 2
    # Apply step two of RSI calc, starting at data point # N+2 (or in index terms, N+1)
    for row in range(N + 1, close.shape[0]):
        avg_up.at[row] = ((avg_up[row - 1] * (N - 1)) + up_dir[row]) / 14
        avg_down.at[row] = ((avg_down[row - 1] * (N - 1)) + down_dir[row]) / 14

    # Calculate relative strength
    RS = avg_up / avg_down

    #Calculate RSI
    RSI = 100 - (100 / (1 + RS))

    #Set RSI to original index
    RSI.index = close.index

    return avg_up, avg_down, RSI

def calc_ema(ohlc: pd.DataFrame, N: int, is_initial_data: bool, EMA_header_name = None):

    '''
    Return the EMA series (exponential moving average) for a given pandas data series of closing price values.

    EMA_today = (Close_today * alpha) + (Close_yesterday * (1 - alpha))

    where:
        alpha = smoothing value / (1 + N)...smoothing value typically = '2'
        N = Number of periods

    Source: https://www.investopedia.com/terms/e/ema.asp

    :param ohlc: (pd.DataFrame) OHLC dataframe input. Contains price values. If data is already initialized, will already have
    the EMA column created.
    :param N: (int) Period value for SMA (simple moving average) and EMA calcs
    :param is_initial_data: (bool) True if initializing OHLC data
    :return: (pd.Series) returns series of EMA values
    '''

    close_series = ohlc.close

    smoothing = 2
    alpha = smoothing / (1 + N)

    if is_initial_data == True:
        EMA_series = pd.Series(index= ohlc.index.to_list(), dtype= 'object')
        EMA_initial_val = close_series [:N].mean()
        EMA_series.iat[N-1] = EMA_initial_val
    else:

        EMA_header = ('EMA' + '_' + str(N)) if EMA_header_name == None else EMA_header_name

        EMA_series = ohlc.loc[:, EMA_header]

    for row in range(N, ohlc.shape[0]):

        EMA_series.iat[row] = (close_series.iat[row] * alpha) + (EMA_series.iat[row-1] * (1 - alpha))

    return EMA_series

def calc_bollinger_bands(ohlc_close: pd.Series, N: int= 20, m: float= 2):

    '''
    Return the bollinger band series for a given data series of closing price values.

    Bollinger bands give an upper and lower threshold in terms of standard deviation from a simple moving average.

    BOLU=MA(n)+m∗σ[TP,n]
    BOLD=MA(n)−m∗σ[TP,n]

    where:
        BOLU=Upper Bollinger Band
        BOLD=Lower Bollinger Band
        MA=Moving average
        n=Number of days in smoothing period (typically 20)
        m=Number of standard deviations (typically 2)
        σ[TP,n]=Standard Deviation over last n periods of TP

    Source: https://www.investopedia.com/terms/b/bollingerbands.asp

    :param ohlc_close: (pd.Series) pandas series of pricing data
    :param N: Number of periods for the SMA (simple moving average..or bollinger middle) and for the standard deviation
    :param m: Multiplier for standard deviation. Bigger value means larger bollinger bands
    :return:
        SMA_Series (pd.Series) Middle of BB range...SMA
        BOLU (pd.Series) Upper threshold of BB range
        BOLD (pd.Series) Lower threshold of BB range
    '''

    SMA_series = ohlc_close.rolling(N).mean()
    std_dev = pd.Series(index= ohlc_close.index.to_list(), dtype= 'object')

    for row in range(N-1, ohlc_close.shape[0]):

        std_dev[row] = ohlc_close[row-(N-1): row+1].std()

    BOLU = SMA_series + (m * std_dev)
    BOLD = SMA_series - (m * std_dev)

    return SMA_series, BOLU, BOLD

def calc_atr(ohlc_df: pd.DataFrame, N: int, is_initial_data: bool, col_name_input: dict= None):

    '''

    https://www.investopedia.com/terms/a/atr.asp
    https://www.thebalance.com/how-average-true-range-atr-can-improve-trading-4154923

    :return:
    '''

    col_name_dict = {'atr': 'average_true_range', 'high': 'high', 'low': 'low', 'close': 'close'} \
        if col_name_input == None else col_name_input

    ohlc = ohlc_df.reset_index(drop= True)

    tr_high_low = ohlc[col_name_dict['high']] - ohlc[col_name_dict['low']]

    if is_initial_data == True:
        tr = pd.Series(data = [np.nan], dtype= float) # Initialize first value as np.nan due to TR formula
        atr = pd.Series(dtype=float)
    else:
        tr = pd.Series(dtype= float)
        # Grab the first N values of the atr. Partly to help deal with the first iteration N values being np.nan
        atr = pd.Series(data = ohlc.loc[0:N, col_name_dict['atr']].to_list())


    for row in range(1, ohlc.shape[0]):

        tr_high_low_val = tr_high_low.at[row]
        tr_high_prevclose = abs(ohlc.at[row, col_name_dict['high']] - ohlc.at[row-1, col_name_dict['close']])
        tr_low_prevclose = abs(ohlc.at[row, col_name_dict['low']] - ohlc.at[row-1, col_name_dict['close']])

        tr.at[row] = max(tr_high_low_val, tr_high_prevclose, tr_low_prevclose)

        # Start at row = N since don't have Tr data point at row = 0
        if row > N: # Do this for both is_initial_data == True or False

            atr.at[row] = (1 / N) * (atr.at[row - 1] * (N - 1) + tr.at[row])

        elif row == N and is_initial_data == True:

            atr.at[row] = (1 / N) * tr[row - N + 1:row + 1].sum()

    return tr, atr

def calc_chandelier_exit(ohlc_df: pd.DataFrame, CE_multiplier: float = 3.5, N: int = 30, is_initial_data: bool = False, col_name_input: dict= None):

    # Allow for manual input of selected dataframe columns. Below are default column names if col_name_input is None
    # Allows function to be applied to alternative candles - ex. Heikin Ashi
    col_name_dict = {'atr': 'average_true_range', 'high': 'high', 'low': 'low', 'close': 'close', 'CE_upper': 'CE_upper', 'CE_lower': 'CE_lower'} \
        if col_name_input == None else col_name_input

    ohlc_df = ohlc_df.reset_index(drop= True)

    N = min(N, ohlc_df.shape[0]-1)

    if is_initial_data == True:
        atr_col = np.where(pd.isnull(ohlc_df[col_name_dict['atr']]), np.nan, ohlc_df[col_name_dict['atr']])
        atr_col_nan = atr_col[np.isnan(atr_col)]
        first_atr_row = len(atr_col_nan)
        CE_upper_list = [0] * len(atr_col_nan)
        CE_lower_list = [0] * len(atr_col_nan)
        first_loop_row = first_atr_row
    else:
        first_atr_row = 0
        CE_upper = ohlc_df.at[0, col_name_dict['CE_upper']]
        CE_lower = ohlc_df.at[0, col_name_dict['CE_lower']]
        CE_upper_list = [CE_upper]
        CE_lower_list = [CE_lower]
        first_loop_row = 1

    for row in range(first_loop_row, ohlc_df.shape[0]):

        highest_high = ohlc_df.loc[row - N + 1: row, col_name_dict['high']].max()
        lowest_low = ohlc_df.loc[row - N + 1: row, col_name_dict['low']].min()
        atr = ohlc_df.at[row, col_name_dict['atr']]
        new_CE_upper = lowest_low + (CE_multiplier * atr)
        new_CE_lower = highest_high - (CE_multiplier * atr)

        prev_CE_upper = CE_upper_list[row - 1]
        prev_CE_lower = CE_lower_list[row - 1]
        prev_close = ohlc_df.at[row - 1, col_name_dict['close']]

        if prev_close > prev_CE_lower:
            CE_lower = max(new_CE_lower, prev_CE_lower)
        else:
            CE_lower = new_CE_lower

        if prev_close < prev_CE_upper:
            CE_upper = min(new_CE_upper, prev_CE_upper)
        else:
            CE_upper = new_CE_upper

        CE_upper_list.append(CE_upper)
        CE_lower_list.append(CE_lower)


    if is_initial_data == True:
        atr_col = np.where(pd.isnull(ohlc_df[col_name_dict['atr']]), np.nan, ohlc_df[col_name_dict['atr']])
        atr_col_nan = atr_col[np.isnan(atr_col)]
        CE_upper_list = list(atr_col_nan) + CE_upper_list[first_atr_row: len(CE_upper_list)] # Remove zeroes and replace w/ np.nan so that it doesn't skew data
        CE_lower_list = list(atr_col_nan) + CE_lower_list[first_atr_row: len(CE_lower_list)] # Remove zeroes and replace w/ np.nan so that it doesn't skew data

    return CE_upper_list, CE_lower_list

def create_trackline(ohlc_df: pd.DataFrame, is_initial_data: bool, col_name_input: dict= None):

    # Allow for manual input of selected dataframe columns. Below are default column names if col_name_input is None
    # Allows function to be applied to alternative candles - ex. Heikin Ashi
    col_name_dict = {'atr': 'average_true_range', 'high': 'high', 'low': 'low', 'close': 'close', 'CE_upper': 'trackline_upper', 'CE_lower': 'trackline_lower', \
                     'trackline': 'trackline', 'trackline_trend': 'trackline_trend'} \
        if col_name_input == None else col_name_input

    ohlc_df = ohlc_df.reset_index(drop=True)
    trackline_upper = ohlc_df[col_name_dict['CE_upper']]
    trackline_lower = ohlc_df[col_name_dict['CE_lower']]
    close = ohlc_df[col_name_dict['close']]

    if is_initial_data or any(np.isnan(ohlc_df[col_name_dict['trackline']]) == False) == False: # If initial data point or no presence of non np.nan values

        is_close_above = np.where(close > trackline_upper, True, False)
        is_close_below = np.where(close < trackline_lower, True, False)

        if True in is_close_above and True in is_close_below:
            first_break_row_upper = list(is_close_above).index(True)
            first_break_row_lower = list(is_close_below).index(True)
            break_row = min(first_break_row_upper, first_break_row_lower)
            start_row = break_row + 1

            if first_break_row_upper < first_break_row_lower:
                first_break_type = 'up'
                first_break_value = trackline_lower.loc[break_row]
            else:
                first_break_type = 'down'
                first_break_value = trackline_upper.loc[break_row]

        elif True in is_close_above:
            break_row = list(is_close_above).index(True)
            start_row = break_row + 1
            first_break_type = 'up'
            first_break_value = trackline_lower.loc[break_row] # If upper trackline broken, use the lower trackline

        elif True in is_close_below:
            break_row = list(is_close_below).index(True)
            start_row = break_row + 1
            first_break_type = 'down'
            first_break_value = trackline_upper.loc[break_row] # If lower trackline broken, use the upper trackline

        else: # If True not in is_close_above and True not in is_close_below:
            start_row = ohlc_df.shape[0]
            break_row = start_row

        trackline = pd.Series(index = range(0, break_row), dtype= float)
        trackline_trend = pd.Series(index = range(0, break_row), dtype= str)

        if True in is_close_above or True in is_close_below:
            trackline = pd.concat([trackline, pd.Series(data=[first_break_value], index=[break_row], dtype= float)])
            trackline_trend = pd.concat([trackline_trend, pd.Series(data=[first_break_type], index=[break_row], dtype= str)])

    else:

        start_row = ohlc_df.shape[0] - 2 # Start 2nd last row
        trackline = ohlc_df[col_name_dict['trackline']]
        trackline_trend = ohlc_df[col_name_dict['trackline_trend']]

    for row in range(start_row, ohlc_df.shape[0]):

        prev_trackline = trackline.loc[row - 1]
        prev_type = trackline_trend.loc[row - 1]
        curr_upper = trackline_upper.loc[row]
        curr_lower = trackline_lower.loc[row]
        curr_close = close.loc[row]

        if prev_type == 'neutral':

            if curr_upper >= prev_trackline and curr_lower >= prev_trackline:
                curr_trend = 'up'
                curr_trackline = trackline_lower.loc[row]

            elif curr_upper <= prev_trackline and curr_lower <= prev_trackline:
                curr_trend = 'down'
                curr_trackline = trackline_upper.loc[row]

            else:
                curr_trend = 'neutral'
                curr_trackline = prev_trackline

        elif prev_type == 'up' and curr_close > curr_lower:
            curr_trend = 'up'
            curr_trackline = trackline_lower.loc[row]

        elif prev_type == 'down' and curr_close < curr_upper:
            curr_trend = prev_type
            curr_trackline = trackline_upper.loc[row]

        else: # Otherwise, close has broken outside of trackline limits so change to neutral

                # prev_type == 'up' and prev_close < prev_trackline or \
                # prev_type == 'down' and prev_close > prev_trackline:

            curr_trend = 'neutral'
            curr_trackline = prev_trackline

        trackline.at[row] = curr_trackline
        trackline_trend.at[row] = curr_trend

    trackline = trackline.to_list()
    trackline_trend = trackline_trend.to_list()

    return trackline, trackline_trend

def find_sr_lines(pair: str, timeframe: int= 1440, num_data_points: int= 60, ohlc_folder_name: str = '../data/ohlc_data'):

    '''

    :param pair:
    :param timeframe: Default 1440 mins (1 day)
    :param num_data_points: Default 60 data points (2 months of daily data points)
    :return:
    '''

    ohlc_folder = ohlc_folder_name
    ohlc_filename = pair + '_' + str(timeframe)

    ohlc_df = read_csv_data(csv_folder_name=ohlc_folder, csv_file_name=ohlc_filename).drop(columns=['Unnamed: 0'])
    size = min(num_data_points, ohlc_df.shape[0])

    lows = ohlc_df.iloc[ohlc_df.shape[0] - size : ohlc_df.shape[0]].low
    highs = ohlc_df.iloc[ohlc_df.shape[0] - size : ohlc_df.shape[0]].high
    lows_and_highs = pd.concat([lows, highs]).to_frame()

    sr_clusters = k_means_get_optimum_clusters(df= lows_and_highs, num_data_points= num_data_points)
    sr_centers = sr_clusters.cluster_centers_
    sr_lines = np.sort(sr_centers, axis=0)

    return sr_lines

def create_heiken_ashi_candles(ohlc_df: pd.DataFrame, is_initial_data: bool):

    ohlc_df = ohlc_df.reset_index(drop = True)

    HA_close = list((0.25) * (ohlc_df.open + ohlc_df.high + ohlc_df.low + ohlc_df.close))

    if is_initial_data == True:
        HA_open = [ohlc_df.open.to_list()[0]]
        HA_high = [max(ohlc_df.high.to_list()[0], HA_open[0], HA_close[0])]
        HA_low = [min(ohlc_df.low.to_list()[0], HA_open[0], HA_close[0])]

    else:
        HA_open = [ohlc_df.HA_open.to_list()[0]]
        HA_high = [ohlc_df.HA_high.to_list()[0]]
        HA_low = [ohlc_df.HA_low.to_list()[0]]

    for row in range(1, ohlc_df.shape[0]):

        HA_open.append((0.50) * (HA_open[row - 1] + HA_close[row - 1]))
        HA_high.append(max(ohlc_df.high.to_list()[row], HA_open[row], HA_close[row]))
        HA_low.append(min(ohlc_df.low.to_list()[row], HA_open[row], HA_close[row]))

    return HA_open, HA_close, HA_high, HA_low

class DefaultLogger:

    def log(self, level, message):
        print(message)

class RetryAndCatch:
    '''
    Source:
    https://codereview.stackexchange.com/questions/133310/python-decorator-for-retrying-w-exponential-backoff
    https://stackoverflow.com/questions/52419395/implementing-a-retry-routine
    '''

    def __init__(self, exceptions_to_catch, delay=0, num_tries=10, logger=DefaultLogger(), log_level=logging.ERROR, logger_attribute=''):
        self.exceptions = exceptions_to_catch
        self.max_tries = num_tries
        self.tries = num_tries
        self.logger = logger
        self.level = log_level
        self.attr_name = logger_attribute
        self.delay = delay
        self.backoff = self.double_backoff

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            backoff_gen = self.backoff(self.delay)
            try:
                while self.tries > 1:
                    try:
                        return f(*args, **kwargs)
                    except self.exceptions as e:
                        message = f"Exception {e} caught, retrying {self.tries - 1} more times."

                        # instance = args[0]
                        # self.logger = getattr(args[0], self.attr_name, self.logger)
                        self.logger.log(self.level, message)

                        time.sleep(self.delay)
                        self.delay = next(backoff_gen)
                        self.tries -= 1

                return f(*args, **kwargs)
            finally:
                self.tries = self.max_tries
        return wrapper

    def double_backoff(self, start):

        if start == 0:
            start = 1
            yield start
        while True:
            start *= 2
            yield start