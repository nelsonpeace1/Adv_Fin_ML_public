# %% [markdown]
# This file contains all the functions implemented through the book.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from typing import Tuple, List, Dict, Union, Optional, Any, Generator

import random

from scipy.stats import rv_continuous, kstest, norm
import scipy.cluster.hierarchy as sch

from datetime import timedelta
from pandas import Timestamp

from sklearn.model_selection._split import _BaseKFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

# %% [markdown]
# # Chapter 2. Financial Data Structures

# %%
def pcaWeights(cov: np.ndarray, riskDist: np.ndarray = None,
               riskTarget: float = 1.) -> np.ndarray:
    eVal, eVec = np.linalg.eigh(cov)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]    # sorting by decreasing eVal (i.e. decreasing variance)
    if riskDist is None:
        riskDist = np.zeros(cov.shape[0])
        riskdist[-1] = 1.
    loads = riskTarget * (riskDist / eVal) ** 0.5
    weights = np.dot(eVec, np.reshape(loads, (-1, 1)))
    return weights

# %%
# symmetrical CUSUM filter
def getTEvents(gRaw: pd.Series, h: float) -> np.ndarray:
    gRaw = gRaw[~gRaw.index.duplicated(keep='first')]
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

# %%
# based on https://towardsdatascience.com/advanced-candlesticks-for-machine-learning-i-tick-bars-a8b93728b4c5
def get_tick_bars(prices: np.ndarray, vols: np.ndarray,
                  times: np.ndarray, freq: int) -> np.ndarray:
    bars = np.zeros(shape=(len(range(freq, len(prices), freq)), 6), dtype=object)
    ind = 0
    for i in range(freq, len(prices), freq):
        bars[ind][0] = pd.Timestamp(times[i - 1])          # time
        bars[ind][1] = prices[i - freq]                    # open
        bars[ind][2] = np.max(prices[i - freq: i])         # high
        bars[ind][3] = np.min(prices[i - freq: i])         # low
        bars[ind][4] = prices[i - 1]                       # close
        bars[ind][5] = np.sum(vols[i - freq: i])           # volume
        ind += 1
    return bars

# %%
def get_volume_bars(prices: np.ndarray, vols: np.ndarray,
                    times: np.ndarray, bar_vol: int) -> np.ndarray:
    bars = np.zeros(shape=(len(prices), 6), dtype=object)
    ind = 0
    last_tick = 0
    cur_volume = 0
    for i in range(len(prices)):
        cur_volume += vols[i]
        if cur_volume >= bar_vol:
            bars[ind][0] = pd.Timestamp(times[i - 1])            # time
            bars[ind][1] = prices[last_tick]                     # open
            bars[ind][2] = np.max(prices[last_tick: i + 1])      # high
            bars[ind][3] = np.min(prices[last_tick: i + 1])      # low
            bars[ind][4] = prices[i]                             # close
            bars[ind][5] = np.sum(vols[last_tick: i + 1])        # volume
            cur_volume = 0
            last_tick = i + 1
            ind += 1
    return bars[:ind]

# %%
def get_dollar_bars(prices: np.ndarray, vols: np.ndarray,
                    times: np.ndarray, bar_sum: int) -> np.ndarray:
    bars = np.zeros(shape=(len(prices), 6), dtype=object)
    ind = 0
    last_tick = 0
    cur_sum = 0
    for i in range(len(prices)):
        cur_sum += vols[i] * prices[i]
        if cur_sum >= bar_sum:
            bars[ind][0] = pd.Timestamp(times[i - 1])            # time
            bars[ind][1] = prices[last_tick]                     # open
            bars[ind][2] = np.max(prices[last_tick: i + 1])      # high
            bars[ind][3] = np.min(prices[last_tick: i + 1])      # low
            bars[ind][4] = prices[i]                             # close
            bars[ind][5] = np.sum(vols[last_tick: i + 1])        # volume
            cur_sum = 0
            last_tick = i + 1
            ind += 1
    return bars[:ind]

# %%
def get_bollinger_bands(dollar_bars: np.ndarray, alpha: float) -> np.ndarray:
    prices = dollar_bars[:, 4]    # taking close prices
    ma = (pd.Series(prices).rolling(20, min_periods=20).mean())      # 20 bars moving average
    sigma = pd.Series(prices).rolling(20, min_periods=20).std()
    b_upper, b_lower = (ma + alpha * sigma), (ma - alpha * sigma)    # bollinger bounds    
    return np.array([ma, b_upper, b_lower])

# %%
def get_returns(bars: np.ndarray) -> np.ndarray:
    close_prices = pd.Series(bars[:, 4], index=bars[:, 0])
    return (close_prices.diff() / close_prices)[1:, ].astype(float)

# %% [markdown]
# # Chapter 3. Labelling

# %%
def get_daily_vol(close: pd.Series, span0: int = 20) -> pd.Series:
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1    # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0

# %%
def apply_tripple_barrier(close: pd.Series, events: pd.DataFrame,
                                   pt_sl: List, molecule: np.ndarray) -> pd.DataFrame:
    '''
    Labeling observations using tripple-barrier method
    
        Parameters:
            close (pd.Series): close prices of bars
            events (pd.DataFrame): dataframe with columns:
                                   - t1: The timestamp of vertical barrier (if np.nan, there will not be
                                         a vertical barrier)
                                   - trgt: The unit width of the horizontal barriers
            pt_sl (list): list of two non-negative float values:
                          - pt_sl[0]: The factor that multiplies trgt to set the width of the upper barrier.
                                      If 0, there will not be an upper barrier.
                          - pt_sl[1]: The factor that multiplies trgt to set the width of the lower barrier.
                                      If 0, there will not be a lower barrier.
            molecule (np.ndarray):  subset of event indices that will be processed by a
                                    single thread (will be used later)
        
        Returns:
            out (pd.DataFrame): dataframe with columns [pt, sl, t1] corresponding to timestamps at which
                                each barrier was touched (if it happened)
    '''
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_['trgt']
    else:
        pt = pd.Series(data=[np.nan] * len(events.index), index=events.index)    # NaNs
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_['trgt']
    else:
        sl = pd.Series(data=[np.nan] * len(events.index), index=events.index)    # NaNs
    
    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc: t1]                                       # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']     # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()        # earlisest stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()        # earlisest profit taking
    return out

# %%
# including metalabeleing possibility
def get_events_tripple_barrier(
    close: pd.Series, tEvents: np.ndarray, pt_sl: float, trgt: pd.Series, minRet: float,
    numThreads: int = 1, t1: Union[pd.Series, bool] = False, side: pd.Series = None
) -> pd.DataFrame:
    '''
    Getting times of the first barrier touch
    
        Parameters:
            close (pd.Series): close prices of bars
            tEvents (np.ndarray): np.ndarray of timestamps that seed every barrier (they can be generated
                                  by CUSUM filter for example)
            pt_sl (float): non-negative float that sets the width of the two barriers (if 0 then no barrier)
            trgt (pd.Series): s series of targets expressed in terms of absolute returns
            minRet (float): minimum target return required for running a triple barrier search
            numThreads (int): number of threads to use concurrently
            t1 (pd.Series): series with the timestamps of the vertical barriers (pass False
                            to disable vertical barriers)
            side (pd.Series) (optional): metalabels containing sides of bets
        
        Returns:
            events (pd.DataFrame): dataframe with columns:
                                       - t1: timestamp of the first barrier touch
                                       - trgt: target that was used to generate the horizontal barriers
                                       - side (optional): side of bets
    '''
    trgt = trgt.loc[trgt.index.intersection(tEvents)]
    trgt = trgt[trgt > minRet]
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    if side is None:
        side_, pt_sl_ = pd.Series(np.array([1.] * len(trgt.index)), index=trgt.index), [pt_sl[0], pt_sl[0]]
    else:
        side_, pt_sl_ = side.loc[trgt.index.intersection(side.index)], pt_sl[:2]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    df0 = apply_tripple_barrier(close, events, pt_sl_, events.index)
#     df0 = mpPandasObj(func=apply_tripple_barrier, pdObj=('molecule', events.index),
#                       numThreads=numThreads, close=close, events=events, pt_sl=[pt_sl, pt_sl])
    events['t1'] = df0.dropna(how='all').min(axis=1)
    if side is None:
        events = events.drop('side', axis=1)
    return events

# %%
def add_vertical_barrier(close: pd.Series, tEvents: np.ndarray, numDays: int) -> pd.Series:
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]])    # adding NaNs to the end
    return t1

# %%
# including metalabeling possibility & modified to generate 0 labels
def get_bins(close: pd.Series, events: pd.DataFrame, t1: Union[pd.Series, bool] = False) -> pd.DataFrame:
    '''
    Generating labels with possibility of knowing the side (metalabeling)
    
        Parameters:
            close (pd.Series): close prices of bars
            events (pd.DataFrame): dataframe returned by 'get_events' with columns:
                                   - index: event starttime
                                   - t1: event endtime
                                   - trgt: event target
                                   - side (optional): position side
            t1 (pd.Series): series with the timestamps of the vertical barriers (pass False
                            to disable vertical barriers)
        
        Returns:
            out (pd.DataFrame): dataframe with columns:
                                       - ret: return realized at the time of the first touched barrier
                                       - bin: if metalabeling ('side' in events), then {0, 1} (take the bet or pass)
                                              if no metalabeling, then {-1, 1} (buy or sell)
    '''
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side']
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0
    else:
        if t1 is not None:
            vertical_first_touch_idx = events_[events_['t1'].isin(t1.values)].index
            out.loc[vertical_first_touch_idx, 'bin'] = 0
    return out

# %%
def drop_labels(labels: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    while True:
        df0 = labels['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        print('dropped label', df0.argmin(), df0.min())
        labels = labels[labels['bin'] != df0.index[df0.argmin()]]
    return labels

# %%
def get_bollinger_bands_df(dollar_bars: pd.DataFrame, alpha: float) -> np.ndarray:
    prices = dollar_bars['close']                                    # taking close prices
    ma = (prices.rolling(30, min_periods=1).mean())                  # 30 bars moving average
    sigma = prices.rolling(30, min_periods=1).std()
    sigma[0] = 0
    b_upper, b_lower = (ma + alpha * sigma), (ma - alpha * sigma)    # bollinger bounds    
    return np.array([ma, b_upper, b_lower])

def get_upside_bars_bb(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df['open'] < df['b_upper']) & (df['close'] > df['b_upper'])]

def get_downside_bars_bb(df: pd.DataFrame) -> np.ndarray:
    return df[(df['open'] > df['b_lower']) & (df['close'] < df['b_lower'])]

# %% [markdown]
# # Chapter 4. Sample Weights

# %%
def num_conc_events(closeIdx: np.ndarray, t1: pd.Series, molecule: np.ndarray) -> pd.Series:
    '''
    Computing the number of concurrent events per bar
    
        Parameters:
            closeIdx (np.ndarray): timestamps of close prices
            t1 (pd.Series): series with the timestamps of the vertical barriers
            molecule (np.ndarray): dates of events on which weights are computed
            
        Returns:
            pd.Series with number of labels concurrent at each timestamp
    '''
    t1 = t1.fillna(closeIdx[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]
    iloc = closeIdx.searchsorted(pd.DatetimeIndex([t1.index[0], t1.max()]))
    count = pd.Series([0] * (iloc[1] + 1 - iloc[0]), index=closeIdx[iloc[0]: iloc[1] + 1])
    for tIn, tOut in t1.iteritems():
        count.loc[tIn: tOut] += 1
    return count.loc[molecule[0]: t1[molecule].max()]

# %%
def sample_weights(t1: pd.Series, num_conc_events: pd.Series, molecule: np.ndarray) -> pd.Series:
    '''
    Computing average uniqueness over the event's lifespan
    
        Parameters:
            t1 (pd.Series): series with the timestamps of the vertical barriers
            num_conc_events (pd.Series): number of concurrent events per bar
            molecule (np.ndarray): dates of events on which weights are computed
            
        Returns:
            weights (pd.Series): weights that represent the average uniqueness
    '''
    weights = pd.Series([0] * len(molecule), index=molecule)
    for tIn, tOut in t1.loc[weights.index].iteritems():
        weights.loc[tIn] = (1.0 / num_conc_events.loc[tIn: tOut]).mean()
    return weights

# %%
def get_ind_matrix(barIdx: np.ndarray, t1: pd.Series) -> pd.DataFrame:
    '''
    Deriving indicator matrix
    
        Parameters:
            barIdx (np.ndarray): indexes of bars
            t1 (pd.Series): series with the timestamps of the vertical barriers
            
        Returns:
            indM (pd.DataFrame): binary matrix indicating what bars influence the label for each observation
    '''
    indM = pd.DataFrame(0, index=barIdx, columns=range(t1.shape[0]))
    for i, (t0, t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1, i] = 1.0
    return indM


def get_avg_uniqueness(indM: pd.DataFrame) -> float:
    '''
    Compute average uniqueness from indicator matrix
    '''
    c = indM.sum(axis=1)
    u = indM.div(c, axis=0)
    avg_uniq = u[u > 0].mean()
    return avg_uniq


def seq_bootstrap(indM: pd.DataFrame, sLength: int = None) -> np.ndarray:
    '''
    Generate a sample via sequential bootstrap
    
        Parameters:
            indM (pd.DataFrame): binary matrix indicating what bars influence the label for each observation
            sLength (int) (optional): sample length (if None, equals number of columns in indM)
            
        Returns:
            phi (np.ndarray): array with indexes of the features sampled by sequential bootstrap
    '''
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avg_uniq = pd.Series()
        for i in indM:
            indM_ = indM[phi + [i]]
            avg_uniq.loc[i] = get_avg_uniqueness(indM_).iloc[-1]
        prob = avg_uniq / avg_uniq.sum()
        phi += [np.random.choice(indM.columns, p=prob)]
    return np.array(phi)

# %%
def gen_rand_t1(numObs: int, numBars: int, maxH: int) -> pd.Series:
    '''
    Generate random t1 series
    
        Parameters:
            numObs (int): number of observations for which t1 is generated
            numBars (int): number of bars
            maxH (int): upper bound for uniform distribution to determine the number of bars spanned by observation
        Returns:
            t1 (pd.Series)
    '''
    t1 = pd.Series()
    for i in range(numObs):
        idx = np.random.randint(0, numBars)
        val = idx + np.random.randint(1, maxH)
        t1.loc[idx] = val
    return t1.sort_index()


def aux_MC(numObs: int, numBars: int, maxH: int) -> dict:
    '''
    Generate random t1 series
    
        Parameters:
            numObs (int): number of observations for which t1 is generated
            numBars (int): number of bars
            maxH (int): upper bound for uniform distribution to determine the number of bars spanned by observation
        Returns:
            dict with average uniqueness derived by standard and sequential bootstrap algorithms
    '''
    t1 = gen_rand_t1(numObs, numBars, maxH)
    barIdx = range(t1.max() + 1)
    indM = get_ind_matrix(barIdx, t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = get_avg_uniqueness(indM[phi]).mean()
    phi = seq_bootstrap(indM)
    seqU = get_avg_uniqueness(indM[phi]).mean()
    return {'stdU': stdU, 'seqU': seqU}


def main_MC(numObs: int, numBars: int, maxH: int, numIters: int) -> None:
    '''
    Run MC simulation for comparing standard and sequential bootstraps
    
        Parameters:
            numObs (int): number of observations for which t1 is generated
            numBars (int): number of bars
            maxH (int): upper bound for uniform distribution to determine the number of bars spanned by observation
            numIters (int): number of MC iterations
        Returns:
            out (pd.DataFrame): dataframe containing uniqueness obtained by standard and sequential bootstraps
    '''
    out = pd.DataFrame()
    for i in range(numIters):
        out = pd.concat((out, pd.DataFrame([aux_MC(numObs, numBars, maxH)])))
    return out

# %%
def sample_return_weights(
    t1: pd.Series, num_conc_events: pd.Series, close: pd.Series, molecule: np.ndarray
) -> pd.Series:
    '''
     Determination of sample weights by absolute return distribution
    
        Parameters:
            t1 (pd.Series): series with the timestamps of the vertical barriers
            num_conc_events (pd.Series): number of concurrent events per bar
            close (pd.Series): close prices
            molecule (np.ndarray): dates of events on which weights are computed
            
        Returns:
            weights (pd.Series): weights that absolute returns
    '''
    ret = np.log(close).diff()
    weights = pd.Series(index=molecule, dtype=object)
    for tIn, tOut in t1.loc[weights.index].iteritems():
        weights.loc[tIn] = (ret.loc[tIn: tOut] / num_conc_events.loc[tIn: tOut]).sum()
    return weights.abs()

# %%
def get_time_decay(tW: pd.Series, clfLastW: float = 1.0) -> pd.Series:
    '''
    Apply piecewise-linear decay to observed uniqueness. Newest observation gets weight=1,
    oldest observation gets weight=clfLastW.
    
        Parameters:
            tW (pd.Series): observed uniqueness
            clfLastW (float): weight for the oldest observation
        
        Returns:
            clfW (pd.Series): series with time-decay factors
    '''
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1. / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1.0 - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0
    return clfW

# %% [markdown]
# # Chapter 5. Fractionally Differentiated Features

# %%
def get_weights(d: float, size: int) -> np.ndarray:
    '''
    Computing the weights for differentiating the series
    
        Parameters:
            d (float): differentiating factor
            size (int): length of weights array
            
        Returns:
            w (np.ndarray): array contatining weights
    '''
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def plot_weights(dRange: list, nPlots: int, size: int) -> None:
    '''
    Generating plots for weights arrays for different differentiating factors
    
        Parameters:
            dRange (list): list with 2 floats - bounds of the interval
            nPlots (int): number of plots
            size(int): length of each weights array
            
        Returns:
            weights (np.ndarray): array contatining weights
    '''
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = get_weights(d, size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how='outer')
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(w)
    ax.set_xlabel('$k$')
    ax.set_ylabel('$w_k$')
    ax.legend(np.round(np.linspace(dRange[0], dRange[1], nPlots), 2), loc='lower right')
    plt.show()

# %%
def frac_diff(series: pd.DataFrame, d: float, thres: float = 0.01) -> pd.DataFrame:
    '''
    Fractional differentiation with increasing width window
    Note 1: For thres=1, nothing is skipped
    Note 2: d can be any positive fractional, not necessarily bounded [0,1]
    
        Parameters:
            series (pd.DataFrame): dataframe with time series
            d (float): differentiating factor
            thres (float): threshold for skipping some of the first observations
        
        Returns:
            df (pd.DataFrame): dataframe with differentiated series
    '''
    w = get_weights(d, series.shape[0])
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), \
                       pd.Series(index=np.arange(series.shape[0]), dtype=object)
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue    # exclude NAs
            df_[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
        df[name] = df_.dropna().copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

# %%
def get_weights_ffd(d: float, thres: float) -> np.ndarray:
    '''
    Computing the weights for differentiating the series with fixed window size
    
        Parameters:
            d (float): differentiating factor
            thres (float): threshold for cutting off weights
            
        Returns:
            w (np.ndarray): array contatining weights
    '''
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

# %%
def frac_diff_ffd(series: pd.DataFrame, d: float, thres: float = 1e-5) -> pd.DataFrame:
    '''
    Fractional differentiation with constant width window
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1]
    
        Parameters:
            series (pd.DataFrame): dataframe with time series
            d (float): differentiating factor
            thres (float): threshold for cutting off weights
        
        Returns:
            df (pd.DataFrame): dataframe with differentiated series
    '''
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), \
                       pd.Series(index=np.arange(series.shape[0]), dtype=object)
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):
                continue    # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.dropna().copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

# %%
def plot_min_ffd(process: Union[np.ndarray, pd.Series, pd.DataFrame],
                 apply_constant_width: bool = True, thres: float = 0.01) -> None:
    '''
    Finding the minimum differentiating factor that passes the ADF test
    
        Parameters:
            process (np.ndarray): array with random process values
            apply_constant_width (bool): flag that shows whether to use constant width window (if True)
                                         or increasing width window (if False)
            thres (float): threshold for cutting off weights
    '''
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf'], dtype=object)
    printed = False
    
    for d in np.linspace(0, 2, 21):
        if apply_constant_width:
            process_diff = frac_diff_ffd(pd.DataFrame(process), d, thres)
        else:
            process_diff = frac_diff(pd.DataFrame(process), d, thres)    
        test_results = adfuller(process_diff, maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(test_results[:4]) + [test_results[4]['5%']]
        if test_results[1] <= 0.05 and not printed:
            print(f'Minimum d required: {d}')
            printed = True
    
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(out['adfStat'])
    ax.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    ax.set_title('Searching for minimum $d$')
    ax.set_xlabel('$d$')
    ax.set_ylabel('ADF statistics')
    plt.show()

# %%
def print_adf_results(process: np.ndarray) -> None:
    '''
    Printing the results of the Augmented Dickey–Fuller test
    '''
    adf, p_value, _, _, _ = adfuller(process, maxlag=1, regression='c', autolag=None)
    print(f'ADF statistics: {adf}')
    print(f'p-value: {p_value}')

# %% [markdown]
# # Chapter 7. Cross-Validation in Finance

# %%
def get_train_times(t1: pd.Series, testTimes: pd.Series) -> pd.Series:
    '''
    Given test times, find the times of the training observations
    
        Parameters:
            t1 (pd.Series): start timestamps (t1.index) and end timestamps (t1.values) of observations
            testTimes (pd.Series): start and end timestamps of testing observations (structure similar to t1)
            
        Returns:
            train (pd.Series): series with purged observations from the training set
    '''
    train = t1.copy(deep=True)
    for start, end in testTimes.iteritems():
        df0 = train[(start <= train.index) & (train.index <= end)].index    # train starts within test
        df1 = train[(start <= train) & (train <= end)].index                # train ends within test
        df2 = train[(train.index <= start) & (end <= train)].index          # train envelops test
        train = train.drop(df0.union(df1).union(df2))
    return train

# %%
def get_embargo_times(times: np.ndarray, pctEmbargo: float = 0.0) -> pd.Series:
    '''
    Get embargo time for each bar
    
        Parameters:
            times (np.ndarray): timestamps of bars
            pctEmbargo (float): share of observations to drop after test
            
        Returns:
            mbrg (pd.Series): series with bar timestamps (mbrg.index) and embargo time for each bar (mbrg.values)
    '''
    step = int(times.shape[0] * pctEmbargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = mbrg.append(pd.Series(times[-1], index=times[-step:]))
    return mbrg

# %%
class PurgedKFold(_BaseKFold):
    '''
    Extend KFold class to work with labels that span intervals.
    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), without training samples in between.
    '''
    
    def __init__(
        self, n_splits: int = 3, t1: Optional[pd.Series] = None, pctEmbargo: float = 0.0
    ) -> None:
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo
        
    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            t0 = self.t1.index[i]    # start of test set
            test_indices = indices[i: j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if maxT1Idx < X.shape[0]:    # right train (with embargo)
                train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
            yield train_indices, test_indices

# %%
def cvScore(
    clf: Any, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series, scoring: str ='neg_log_loss',
    t1: Optional[pd.Series] = None, cv: Optional[int] = None,
    cvGen: Optional[PurgedKFold] = None, pctEmbargo: Optional[float] = None
) -> np.ndarray:
    '''
    Calculating cross-validation score.
    
        Parameters:
            clf (Any): model we want to fit
            X (pd.DataFrame): feature matrix
            y (pd.Series): labels
            sample_weight (pd.Series): sample weights
            scoring (str): score we want to compute
            t1 (pd.Series): start timestamps (t1.index) and end timestamps (t1.values) of observations
            cv (int): number of splits
            cvGen (PurgedKFold): object of PurgedKFold class to make splitting
            pctEmbargo (float): share of observations to drop after test
            
        Returns:
            score (np.ndarray): score for each cross-validation split
    '''
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method')
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)    # purged
    score = []
    for train, test in cvGen.split(X=X):
        fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob, sample_weight=sample_weight.iloc[test].values, labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)

# %%
def plot_cv_results(cv: Union[StratifiedKFold, PurgedKFold], clf: Any, X: pd.DataFrame, y: pd.Series) -> None:
    '''
    Plots ROC curve for each iteration of cross-validation together with the mean curve
    and print cv accuracy.
    Based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval
    '''
    for scoring in ['accuracy', 'precision', 'recall']:
        score = cross_val_score(estimator=clf, X=X, y=y, scoring=scoring, cv=cv, n_jobs=-1)
        print(f'CV mean {scoring}: {np.mean(score)}')
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (train, test) in enumerate(cv.split(X, y)):
        clf.fit(np.array(X)[train], np.array(y)[train])
        viz = RocCurveDisplay.from_estimator(clf, np.array(X)[test], np.array(y)[test], name="ROC fold {}".format(i),
                                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="b", label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2, alpha=0.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
    ax.legend(loc="lower right")
    plt.show()

# %% [markdown]
# # Chapter 8. Feature Importance

# %%
def feat_imp_MDI(fit: Any, featNames: np.ndarray) -> pd.DataFrame:
    '''
    Calculates mean feature importances based on MDI.
    
        Parameters:
            fit (Any): classifier (needs to be tree-based, e.g. Random Forest)
            featNames (np.ndarray): list with feature names
        
        Returns:
            imp (pd.DataFrame): dataframe with mean and std of importance for each feature
    '''
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)    # because max_features=1
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std() * df0.shape[0] ** (-0.5)}, axis=1)
    imp /= imp['mean'].sum()
    return imp

# %%
def feat_imp_MDA(
    clf: Any, X: pd.DataFrame, y: pd.Series, cv: int, sample_weight: pd.Series,
    t1: pd.Series, pctEmbargo: float, scoring: str = 'neg_log_loss'
) -> Tuple[pd.DataFrame, float]:
    '''
    Calculates mean feature importances based on OOS score reduction
    while also fitting and evaluating classifier.
    
        Parameters:
            clf (Any): model we want to fit
            X (pd.DataFrame): feature matrix
            y (pd.Series): labels
            cv (int): number of splits
            sample_weight (pd.Series): sample weights
            t1 (pd.Series): start timestamps (t1.index) and end timestamps (t1.values) of observations
            pctEmbargo (float): share of observations to drop after test
            scoring (str): score we want to compute
        
        Returns:
            imp (pd.DataFrame): dataframe with mean and std of importance for each feature
            scr0.mean() (float): mean CV score of classifier
    '''
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')
    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)    # purged cv
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns, dtype=object)
    
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)    # permutation of a single column
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)
    imp = (-scr1).add(scr0, axis=0)
    if scoring == 'neg_log_loss':
        imp = imp / -scr1
    else:
        imp = imp / (1.0 - scr1)
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** (-0.5)}, axis=1)
    return imp, scr0.mean()

# %%
def aux_feat_imp_SFI(
    featNames: np.ndarray, clf: Any, trnsX: pd.DataFrame, cont: pd.DataFrame, scoring: str, cvGen: PurgedKFold
) -> pd.DataFrame:
    '''
    Calculates mean feature importances based on Single Feature Importance (SFI).
    
        Parameters:
            featNames (np.ndarray): list with feature names
            clf (Any): model we want to fit
            trnsX (pd.DataFrame): train dataset
            cont (pd.DataFrame): dataframe with observation labels and weights
            scoring (str): scoring function used for evaluation
            cvGen (PurgedKFold): CV generator (purged)
        
        Returns:
            imp (pd.DataFrame): dataframe with mean and std of importance for each feature
    '''
    imp = pd.DataFrame(columns=['mean', 'std'], dtype=object)
    for featName in featNames:
        df0 = cvScore(clf, X=trnsX[[featName]], y=cont['bin'], sample_weight=cont['w'], scoring=scoring, cvGen=cvGen)
        imp.loc[featName, 'mean'] = df0.mean()
        imp.loc[featName, 'std'] = df0.std() * df0.shape[0] ** (-0.5)
    return imp

# %%
def get_eVec(dot: np.ndarray, varThres: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculates eigenvalues and eigenvectors of dot product matrix that explain varThres of its variance.
    
        Parameters:
            dot (np.ndarray): feature matrix
            varThres (float): share of variance we want to explain
        
        Returns:
            eVal (np.ndarray): eigenvalues
            eVec (np.ndarray): eigenvectors
    '''
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]    # arguments for sorting eVal desc
    eVal, eVec = eVal[idx], eVec[:, idx]
    eVal = pd.Series(eVal, index=['PC_' + str(i + 1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    eVec = eVec.loc[:, eVal.index]
    cumVar = eVal.cumsum() / eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal, eVec = eVal.iloc[:dim + 1], eVec.iloc[:, :dim + 1]
    return eVal, eVec


def ortho_feats(dfX: pd.DataFrame, varThres: float = 0.95) -> pd.DataFrame:
    '''
    Given a dataframe dfX of features, compute orthogonal features dfP explaining varThres of variance.
    
        Parameters:
            dfX (pd.DataFrame): feature matrix
            varThres (float): share of variance we want to explain
        
        Returns:
            dfP (pd.DataFrame): orthogonal features
    '''
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)    # standardize
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    eVal, eVec = get_eVec(dot, varThres)
    dfP = np.dot(dfZ, eVec)
    return pd.DataFrame(dfP)

# %%
def get_test_data(n_features: int = 40, n_informative: int = 10,
                  n_redundant: int = 10, n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Generate a synthetic dataset with given types of features.
    
        Parameters:
            n_features (int): total number of features
            n_informative (int): number of informative features
            n_redundant (int): number of redundant features (linear combinations of informative features)
            n_samples (int): number of observations
        
        Returns:
            trnsX (pd.DataFrame): synthetic dataset
            cont (pd.DataFrame): dataframe with labels ('bin'), weights, and t1 timestamps
    '''
    trnsX, cont = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                      n_redundant=n_redundant, random_state=0, shuffle=False)
    trnsX, cont = pd.DataFrame(trnsX), pd.Series(cont).to_frame('bin')
    df0 = ['I_' + str(i) for i in range(n_informative)] + ['R_' + str(i) for i in range(n_redundant)]
    df0 += ['N_' + str(i) for i in range(n_features - len(df0))]
    trnsX.columns = df0
    cont['w'] = 1.0 / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)
    return trnsX, cont

# %%
# no multiprocessing here
def feat_importance(
    trnsX: pd.DataFrame, cont: pd.DataFrame, n_estimators: int = 100, cv: int = 10,
    max_samples: float = 1.0, pctEmbargo: float = 0.0, scoring: str = 'accuracy',
    method: str = 'SFI', min_weight_fraction_leaf: float = 0.0, ensemble: str = 'bagging'
) -> Tuple[pd.DataFrame, float, float]:
    '''
    Calculate feature importance using given method using bagged decision trees.
    
        Parameters:
            ensemble (str): model type (decision trees bagging or random forest)
            trnsX (pd.DataFrame): train dataset
            cont (pd.DataFrame): dataframe with labels ('bin'), weights, and t1 timestamps
            n_estimators (int): number of trees
            cv (int): number of CV splits
            max_samples (float): share of samples to draw from X to train each base estimator
            pctEmbargo (float): share of observations to drop after test (embargo period)
            scoring (str): scoring/loss function
            method (str): method used to calculate feature importance
            min_weight_fraction_leaf (float): minimum fraction of the sum of weights required to be at a leaf node
        
        Returns:
            imp (pd.DataFrame): dataframe with mean and std of importance for each feature
            oob (float): out-of-bag classifier score
            oos (float): mean CV score
    '''
    if ensemble == 'bagging':
        clf = DecisionTreeClassifier(criterion='entropy', max_features=1, class_weight='balanced',
                                     min_weight_fraction_leaf=min_weight_fraction_leaf)
        clf = BaggingClassifier(base_estimator=clf, n_estimators=n_estimators, max_features=1.0,
                                max_samples=max_samples, oob_score=True, n_jobs=-1)
    else:
        clf = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy',
                                     min_weight_fraction_leaf=min_weight_fraction_leaf,
                                     max_features=1, oob_score=True, n_jobs=-1, max_samples=max_samples)
    fit = clf.fit(X=trnsX, y=cont['bin'], sample_weight=cont['w'].values)
    oob = fit.oob_score_
    
    if method == 'MDI':
        imp = feat_imp_MDI(fit, featNames=trnsX.columns)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'], t1=cont['t1'],
                      pctEmbargo=pctEmbargo, scoring=scoring).mean()
        
    elif method=='MDA':
        imp, oos = feat_imp_MDA(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'], t1=cont['t1'],
                                pctEmbargo=pctEmbargo, scoring=scoring)
        
    elif method=='SFI':
        cvGen = PurgedKFold(n_splits=cv, t1=cont['t1'], pctEmbargo=pctEmbargo)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], sample_weight=cont['w'],
                      scoring=scoring, cvGen=cvGen).mean()
        imp = aux_feat_imp_SFI(featNames=trnsX.columns, clf=clf, trnsX=trnsX, cont=cont,
                               scoring=scoring, cvGen=cvGen)
    
    return imp, oob, oos

# %%
def plot_feat_importance(
    imp: pd.DataFrame, oob: float, oos: float, method: str,
    tag: str = 'test_func', simNum: Optional[str] = None
) -> None:
    '''
    Plots mean feature importance bars with std.
    
        Parameters:
            imp (pd.DataFrame): feature importance
            oob (float): out-of-bag score
            oos (float): mean CV score
            method (str): method to calculate feature importance
            tag (str): tag for title
            simNum (str): reference for simulation parameters
    '''
    fig, ax = plt.subplots(figsize=(10, imp.shape[0] / 5))
    imp = imp.sort_values('mean', ascending=True)
    ax.barh(y=imp.index, width=imp['mean'], color='b', alpha=0.25, xerr=imp['std'], error_kw={'ecolor':'r'})
    if method=='MDI':
        ax.set_xlim(left=0, right=imp.sum(axis=1).max())
        ax.axvline(1.0 / imp.shape[0], linewidth=1, color='r', linestyle='dotted')
    ax.get_yaxis().set_visible(False)
    for i, j in zip(ax.patches, imp.index):
        ax.text(i.get_width() / 2, i.get_y() + i.get_height() / 2, j, ha='center', va='center', color='black')
    ax.set_title(f'tag={tag} | simNum={simNum} | oob={str(round(oob, 4))} | oos={str(round(oos,4))}')
    plt.show()

# %%
def test_func(n_features: int = 40, n_informative: int = 10, n_redundant: int = 10,
              n_estimators: int = 100, n_samples: int = 10000, cv: int = 10) -> pd.DataFrame:
    '''
    Run 3 methods to calculate feature importance on synthetic dataset and print the results.
    
        Parameters:
            n_features (int): total number of features
            n_informative (int): number of informative features
            n_redundant (int): number of redundant features (linear combinations of informative features)
            n_estimators (int): number of trees
            n_samples (int): number of observations
            cv (int): number of CV splits
        
        Returns:
            out (pd.DataFrame): dataframe with stats on each method
    '''
    trnsX, cont = get_test_data(n_features, n_informative, n_redundant, n_samples)
    dict0 = {'minWLeaf': [0.0], 'scoring': ['accuracy'], 'method': ['MDI', 'MDA', 'SFI'], 'max_samples': [1.0]}
    jobs, out = (dict(zip(dict0, i)) for i in product(*dict0.values())), []
    
    for job in jobs:
        job['simNum'] = job['method'] +'_' + job['scoring'] + '_' + '%.2f'%job['minWLeaf'] + \
                        '_' + str(job['max_samples'])
        print(job['simNum'])
        imp, oob, oos = feat_importance(trnsX=trnsX, cont=cont, n_estimators=n_estimators,
                                        cv=cv, max_samples=job['max_samples'], scoring=job['scoring'],
                                        method=job['method'])
        plot_feat_importance(imp=imp, oob=oob, oos=oos, method=job['method'],
                             tag='test_func', simNum=job['simNum'])
        df0 = imp[['mean']] / imp['mean'].abs().sum()
        df0['type'] = [i[0] for i in df0.index]
        df0 = df0.groupby('type')['mean'].sum().to_dict()
        df0.update({'oob': oob, 'oos': oos})
        df0.update(job)
        out.append(df0)
    
    out = pd.DataFrame(out).sort_values(['method', 'scoring', 'minWLeaf', 'max_samples'])
    out = out[['method', 'scoring', 'minWLeaf', 'max_samples', 'I', 'R', 'N', 'oob', 'oos']]
    return out

# %% [markdown]
# # Chapter 9. Hyper-Parameter Tuning with Cross-Validation

# %%
class MyPipeline(Pipeline):
    '''
    Augmentation of sklearn Pipeline class that allows to pass 'sample_weight' to 'fit' method.
    '''
    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None, **fit_params
    ) -> 'MyPipeline':
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super().fit(X, y, **fit_params)

# %%
def clf_hyper_fit_base(
    feat: pd.DataFrame, lbl: pd.Series, t1: pd.Series, pipe_clf: Any, param_grid: Dict[str, list],
    cv: int = 3, bagging: list = [0, None, 1.0], n_jobs: int = -1, pctEmbargo: float = 0.0, **fit_params
) -> Any:
    '''
    Implements purged GridSearchCV with a possibility of fitting bagging of tuned estimator.
    
        Parameters:
            feat (pd.DataFrame): features dataset
            lbl (pd.Series): labels
            t1 (pd.Series): start timestamps (t1.index) and end timestamps (t1.values) of observations
            pipe_clf (Any): classififer to fit
            param_grid (Dict[str, list]): dictionary with parameters values
            cv (int): number of splits
            bagging (list): bagging parameters (used when bagging[1] is not None)
            n_jobs (int): number of jobs to run in parallel
            pctEmbargo (float): share of observations to drop after train
        
        Returns:
            gs (Any): fitted best estimator found by grid search
    '''
    if set(lbl.values) == {0, 1}:
        scoring='f1'    # f1 for meta-labeling
    else:
        scoring='neg_log_loss'    # symmetric towards all cases
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)    # purged
    gs=GridSearchCV(estimator=pipe_clf ,param_grid=param_grid, scoring=scoring, cv=inner_cv, n_jobs=n_jobs)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_    # pipeline
    if bagging[1] is not None and bagging[1] > 0:
        gs = BaggingClassifier(base_estimator=MyPipeline(gs.steps), n_estimators=int(bagging[0]),
                               max_samples=float(bagging[1]), max_features=float(bagging[2]), n_jobs=n_jobs)
        gs = gs.fit(feat, lbl, sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs

# %%
# expand clf_hyper_fit_base to incorporate option to use randomized grid search
def clf_hyper_fit(
    feat: pd.DataFrame, lbl: pd.Series, t1: pd.Series, pipe_clf: Any, param_grid: Dict[str, list],
    cv: int = 3, bagging: list = [0, None, 1.0], rndSearchIter: int = 0,
    n_jobs: int = -1, pctEmbargo: float = 0.0, **fit_params
) -> Any:
    '''
    Implements purged GridSearchCV with a possibility of fitting bagging of tuned estimator.
    
        Parameters:
            feat (pd.DataFrame): features dataset
            lbl (pd.Series): labels
            t1 (pd.Series): start timestamps (t1.index) and end timestamps (t1.values) of observations
            pipe_clf (Any): classififer to fit
            param_grid (Dict[str, list]): dictionary with parameters values
            cv (int): number of splits
            bagging (list): bagging parameters (used when bagging[1] is not None)
            rndSearchIter (int): number of iterations to use in randomized GS (if 0 then apply standard GS)
            n_jobs (int): number of jobs to run in parallel
            pctEmbargo (float): share of observations to drop after train
        
        Returns:
            gs (Any): fitted best estimator found by grid search
    '''
    if set(lbl.values) == {0, 1}:
        scoring='f1'    # f1 for meta-labeling
    else:
        scoring='neg_log_loss'    # symmetric towards all cases
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)    # purged
    
    if rndSearchIter == 0:
        gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid, scoring=scoring, cv=inner_cv, n_jobs=n_jobs)
    else:
        gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid, scoring=scoring,
                                cv=inner_cv, n_jobs=n_jobs, n_iter=rndSearchIter)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_    # pipeline
    
    if bagging[1] is not None and bagging[1] > 0:
        gs = BaggingClassifier(base_estimator=MyPipeline(gs.steps), n_estimators=int(bagging[0]),
                               max_samples=float(bagging[1]), max_features=float(bagging[2]), n_jobs=n_jobs)
        gs = gs.fit(feat, lbl, sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs

# %%
class logUniform_gen(rv_continuous):
    '''
    Implements generator of log-uniform random variables.
    '''
    def _cdf(self, x: float) -> float:
        return np.log(x / self.a) / np.log(self.b / self.a)


def log_uniform(a: float = 1.0, b: float = np.exp(1.0)) -> 'logUniform_gen':
    return logUniform_gen(a=a, b=b, name='log_uniform')

# %%
def get_IS_sharpe_ratio(clf: Any) -> float:
    '''
    Given a fitted gridsearch classifier, returns Sharpe ratio of the best estimator's in-sample forecasts.
    '''
    best_estimator_ind = np.argmin(clf.cv_results_['rank_test_score'])
    mean_score = clf.cv_results_['mean_test_score'][best_estimator_ind]
    std_score = clf.cv_results_['std_test_score'][best_estimator_ind]
    if mean_score < 0:
        return -mean_score / std_score
    else:
        return mean_score / std_score

# %% [markdown]
# # Chapter 10. Bet Sizing

# %%
def avg_active_signals_(signals: pd.DataFrame, molecule: np.ndarray) -> pd.Series:
    '''
    Auxilary function for averaging signals. At time loc, averages signal among those still active.
    Signal is active if:
        a) issued before or at loc AND
        b) loc before signal's endtime, or endtime is still unknown (NaT).
    
        Parameters:
            signals (pd.DataFrame): dataset with signals and t1
            molecule (np.ndarray): dates of events on which weights are computed
        
        Returns:
            out (pd.Series): series with average signals for each timestamp
    '''
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.values <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
        act = signals[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            out[loc] = 0    # no signals active at this time
    return out
            

def avg_active_signals(signals: pd.DataFrame) -> pd.Series:
    '''
    Computes the average signal among those active.
    
        Parameters:
            signals (pd.DataFrame): dataset with signals and t1
        
        Returns:
            out (pd.Series): series with average signals for each timestamp
    '''
    tPnts = set(signals['t1'].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = sorted(list(tPnts))
    out = avg_active_signals_(signals=signals, molecule=tPnts)
    return out

# %%
def discrete_signal(signal0: pd.Series, stepSize: float) -> pd.Series:
    '''
    Discretizes signals.
    
        Parameters:
            signal0 (pd.Series): series with signals
            stepSize (float): degree of discretization (must be in (0, 1])
        
        Returns:
            signal1 (pd.Series): series with discretized signals
    '''
    signal1 = (signal0 / stepSize).round() * stepSize    # discretize
    signal1[signal1 > 1] = 1    # cap
    signal1[signal1 < -1] = -1    # floor
    return signal1

# %%
# no multithreading
def get_signal(
    events: pd.DataFrame, stepSize: float, prob: pd.Series, pred: pd.Series, numClasses: int, **kargs
) -> pd.Series:
    '''
    Gets signals from predictions. Includes averaging of active bets as well as discretizing final value.
    
        Parameters:
            events (pd.DataFrame): dataframe with columns:
                                       - t1: timestamp of the first barrier touch
                                       - trgt: target that was used to generate the horizontal barriers
                                       - side (optional): side of bets
            stepSize (float): ---
            prob (pd.Series): series with probabilities of given predictions
            pred (pd.Series): series with predictions
            numClasses (int): number of classes
        
        Returns:
            signal1 (pd.Series): series with discretized signals
    '''
    if prob.shape[0] == 0:
        return pd.Series()
    signal0 = (prob - 1.0 / numClasses) / (prob * (1.0 - prob)) ** 0.5    # t-value
    signal0 = pred * (2 * norm.cdf(signal0) - 1)    # signal = side * size
    if 'side' in events:
        signal0 *= events.loc[signal0.index, 'side']    # meta-labeling
    df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
    df0 = avg_active_signals(df0)
    signal1 = discrete_signal(signal0=df0, stepSize=stepSize)
    return signal1

# %%
def bet_size(x: float, w: float) -> float:
    '''
    Returns bet size given price divergence and sigmoid function coefficient.
    
        Parameters:
            x (float): difference between forecast price and current price f_i - p_t
            w (float): coefficient that regulates the width of the sigmoid function
        
        Returns:
            (float): bet size
    '''
    return x * (w + x ** 2) ** (-0.5)


def get_target_pos(w: float, f: float, mP: float, maxPos: float) -> float:
    '''
    Calculates target position size associated with forecast f.
    
        Parameters:
            w (float): coefficient that regulates the width of the sigmoid function
            f (float): forecast price
            mP (float): current market price
            maxPos (float): maximum absolute position size
        
        Returns:
            (float): target position size
    '''
    return int(bet_size(w, f - mP) * maxPos)


def inv_price(f: float, w: float, m: float) -> float:
    '''
    Calculates inverse function of bet size with respect to market price p_t.
    
        Parameters:
            f (float): forecast price
            w (float): coefficient that regulates the width of the sigmoid function
            m (float): bet size
            
        Returns:
            (float): inverse price function
    '''
    return f - m * (w / (1 - m**2)) ** 0.5


def limit_price(tPos: float, pos: float, f: float, w: float, maxPos: float) -> float:
    '''
    Calculates breakeven limit price p̄ for the order size q̂_{i,t} − q_t to avoid realizing losses.
    
        Parameters:
            tPos (float): target position
            pos (float): current position
            f (float): forecast price
            w (float): coefficient that regulates the width of the sigmoid function
            maxPos (float): maximum absolute position size
        
        Returns:
            lP (float): limit price
    '''
    sgn = (1 if tPos >= pos else -1)
    lP = 0
    for j in range(abs(pos + sgn), abs(tPos + 1)):
        lP += inv_price(f, w, j / float(maxPos))
    lP /= tPos - pos
    return lP


def get_w(x: float, m: float):
    '''
    Calibrates sigmoid coefficient by calculating the inverse function of bet size with respect to w.
    
        Parameters:
            x (float): difference between forecast price and current price f_i - p_t
            m (float): bet size
    '''
    return x ** 2 * (m**(-2) - 1)

# %%
def get_num_conc_bets_by_date(date: Timestamp, signals: pd.DataFrame) -> Tuple[int, int]:
    '''
    Derives number of long and short concurrent bets by given date.
    
        Parameters:
            date (Timestamp): date of signal
            signals (pd.DataFrame): dataframe with signals
            
        Returns:
            long, short (Tuple[int, int]): number of long and short concurrent bets
    '''
    long, short = 0, 0
    for ind in pd.date_range(start=max(signals.index[0], date - timedelta(days=25)), end=date):
        if ind <= date and signals.loc[ind]['t1'] >= date:
            if signals.loc[ind]['signal'] >= 0:
                long += 1
            else:
                short += 1
    return long, short

# %% [markdown]
# # Chapter 13. Backtesting on Synthetic Data

# %%
def batch(
    coeffs: Dict[str, float], nIter: int = 1e4, maxHP: int = 100, rPT: np.ndarray = np.linspace(0.5, 10, 20),
    rSLm: np.ndarray = np.linspace(0.5, 10, 20), seed: float = 0.0
) -> list:
    '''
    Computes a 20×20 mesh of Sharpe ratios, one for each trading rule, given a pair of initial parameters.
    
        Parameters:
            coeffs (dict): dictionary with values of forecast price, half-life of the process and sigma parameter
            nIter (int): number of paths to simulate
            maxHP (int): maximum holding period
            rPT (np.ndarray): profit take upper bound (in std units)
            rSLm (np.ndarray): stop loss lower bound (in std units)
            seed (float): initial price P_{i, t} (can be fixed to 0, drives the converges only and not absolute values)
            
        Returns:
            output1 (list): array contatining bounds combination and strategy performance
    '''
    phi, output1 = 2 ** (-1.0 / coeffs['hl']), []
    for comb_ in product(rPT, rSLm):
        output2 = []
        for iter_ in range(int(nIter)):
            p, hp, count = seed, 0, 0
            while True:
                p = (1 - phi) * coeffs['forecast'] + phi * p + coeffs['sigma'] * gauss(0, 1)
                cP = p - seed
                hp += 1
                if cP > comb_[0] or cP < -comb_[1] or hp > maxHP:
                    output2.append(cP)
                    break
        mean, std = np.mean(output2), np.std(output2)
        output1.append((comb_[0], comb_[1], mean, std, mean / std))
    return output1

# %%
def iterate_parameters():
    '''
    Iterates over different combinations of initial parameters and run simulations for each.
    '''
    rPT = rSLm = np.linspace(0,10,21)
    count = 0
    for prod_ in product([10, 5, 0, -5, -10], [5, 10, 25, 50, 100]):
        count += 1
        coeffs = {'forecast': prod_[0], 'hl': prod_[1], 'sigma': 1}
        output = batch(coeffs, nIter=1e4, maxHP=100, rPT=rPT, rSLm=rSLm)
    return output

# %% [markdown]
# # Chapter 14. Backtest Statistics

# %%
def get_bets_timing(tPos: pd.Series) -> pd.Index:
    '''
    Calculates the timestamps of flattening or flipping trades from target positions series.
    
    Parameters:
        tPos (pd.Series): series with target positions
        
    Returns:
        bets (pd.Index): bets timing
    '''
    df0 = tPos[tPos == 0].index
    df1 = tPos.shift(1)
    df1 = df1[df1 != 0].index
    bets = df0.intersection(df1)    # flattening
    df0 = tPos.iloc[1:] * tPos.iloc[:-1].values
    bets = bets.union(df0[df0 < 0].index).sort_values()    # tPos flips
    if tPos.index[-1] not in bets:
        bets = bets.append(tPos.index[-1:])    # last bet
    return bets

# %%
def get_holding_period(tPos: pd.Series) -> float:
    '''
    Derives average holding period (in days) using average entry time pairing algo.
    
    Parameters:
        tPos (pd.Series): series with target positions
        
    Returns:
        hp (float): holding period
    '''
    hp, tEntry = pd.DataFrame(columns=['dT', 'w']), 0.0
    pDiff, tDiff = tPos.diff(), (tPos.index - tPos.index[0]) / np.timedelta64(1, 'D')
    for i in range(1, tPos.shape[0]):
        if pDiff.iloc[i] * tPos.iloc[i - 1] >= 0:    # increased or unchanged
            if tPos.iloc[i] != 0:
                tEntry = (tEntry * tPos.iloc[i - 1] + tDiff[i] * pDiff.iloc[i]) / tPos.iloc[i]
        else:    # decreased
            if tPos.iloc[i] * tPos.iloc[i-1] < 0:    # flip
                hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(tPos.iloc[i - 1]))
                tEntry = tDiff[i]    # reset entry time
            else:
                hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(pDiff.iloc[i]))
    if hp['w'].sum() > 0:
        hp = (hp['dT'] * hp['w']).sum() / hp['w'].sum()
    else:
        hp = np.nan
    return hp

# %%
def get_HHI(betRet: pd.Series) -> float:
    '''
    Derives HHI concentration of returns (see p. 200 for definition). Returns can be divided into positive
    and negative or you can calculate the concentration of bets across the months.
    
    Parameters:
        betRet (pd.Series): series with bets returns
        
    Returns:
        hhi (float): concentration
    '''
    if betRet.shape[0] <= 2:
        return np.nan
    wght = betRet / betRet.sum()
    hhi = (wght ** 2).sum()
    hhi = (hhi - betRet.shape[0] ** (-1)) / (1.0 - betRet.shape[0] ** (-1))
    return hhi

# %%
def compute_DD_TuW(series: pd.Series, dollars: bool = False) -> Tuple[pd.Series, pd.Series]:
    '''
     Computes series of drawdowns and the time under water associated with them.
    
    Parameters:
        series (pd.Series): series with either returns (dollars=False) or dollar performance (dollar=True)
        dollars (bool): indicator charachterizing series
        
    Returns:
        dd (pd.Series): drawdown series
        tuw (pd.Series): time under water series
    '''
    df0 = series.to_frame('pnl')
    df0['hwm'] = series.expanding().max()
    df1 = df0.groupby('hwm').min().reset_index()
    df1.columns = ['hwm', 'min']
    df1.index = df0['hwm'].drop_duplicates(keep='first').index    # time of hwm
    df1 = df1[df1['hwm'] > df1['min']]    # hwm followed by a drawdown
    if dollars:
        dd = df1['hwm'] - df1['min']
    else:
        dd = 1 - df1['min'] / df1['hwm']
    tuw = ((df1.index[1:] - df1.index[:-1]) / np.timedelta64(1, 'Y')).values    # in years
    tuw = pd.Series(tuw, index=df1.index[:-1])
    return dd, tuw

# %% [markdown]
# # Chapter 15. Understanding Strategy Risk

# %%
def estimate_SR(prob: float, sl: float, pt: float, freq: float, num_trials: int = 1000000) -> float:
    '''
    Estimates strategy's Sharpe ratio under given parameters.
    
        Parameters:
            prob (float): precision of the strategy
            sl (float): stop loss threshold
            pt (float): profit taking threshold
            freq (float): annual number of bets (to obtain annualized SR)
            num_trial (int): number of trials used for estimation
            
        Returns:
            sr (float): Sharpe ratio
    '''
    out = []
    for i in range(num_trials):
        rnd = np.random.binomial(n=1, p=prob)
        if rnd == 1:
            x = pt
        else:
            x = sl
        out.append(x)
    sr = np.mean(out) / np.std(out) * np.sqrt(freq)
    return sr

# %%
def bin_HR(sl: float, pt: float, freq: float, tSR: float) -> float:
    '''
    Returns minimum precision p needed to achieve target Sharpe ration under given parameters.
    
        Parameters:
            sl (float): stop loss threshold
            pt (float): profit taking threshold
            freq (float): annual number of bets
            tSR (float): target annual Sharpe ratio
            
        Returns:
            p (float): precision
    '''
    a = (freq + tSR ** 2) * (pt - sl) ** 2
    b = (2 * freq * sl - tSR ** 2 * (pt - sl)) * (pt - sl)
    c = freq * sl ** 2
    p = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2.0 * a)
    return p

# %%
def bin_freq(sl: float, pt: float, p: float, tSR: float) -> float:
    '''
    Returns minimum number of bets per year needed to achieve target Sharpe ration under given parameters.
    
        Parameters:
            sl (float): stop loss threshold
            pt (float): profit taking threshold
            p (float): precision
            tSR (float): target annual Sharpe ratio
            
        Returns:
            freq (float): annual number of bets
    '''
    freq = (tSR * (pt - sl)) ** 2 * p * (1 - p) / ((pt - sl) * p + sl) ** 2
    return freq

# %%
def mix_gaussians(
    mu1: float, mu2: float, sigma1: float, sigma2: float, prob1: float, nObs: int
) -> np.ndarray:
    '''
    Generates random draws form a mixture of two Gaussians.
    
        Parameters:
            mu1 (float): expectation of 1st Gaussian
            mu2 (float): expectation of 2nd Gaussian
            sigma1 (float): std of 1st Gaussian
            sigma2 (float): std of 2nd Gaussian
            prob1 (float): probability of generating from 1st Gaussian (i.e. weight of 1st Gaussian)
            nObs (int): total number of draws
            
        Returns:
            ret (np.ndarray): array with observations
    '''
    ret1 = np.random.normal(mu1, sigma1, size=int(nObs * prob1))
    ret2 = np.random.normal(mu2, sigma2, size=nObs - ret1.shape[0])
    ret = np.append(ret1, ret2, axis=0)
    np.random.shuffle(ret)
    return ret

# %%
def prob_failure(ret: np.ndarray, freq: float, tSR: float):
    '''
    Derives probability that strategy has lower precision than needed.
    
        Parameters:
            ret (np.ndarray): array with observations
            freq (float): annual number of bets
            tSR (float): target Sharpe ratio
            
        Returns:
            risk (float): probability of failure
    '''
    rPos, rNeg = ret[ret > 0].mean(), ret[ret <= 0].mean()
    p = ret[ret > 0].shape[0] / float(ret.shape[0])
    thresP = bin_HR(rNeg, rPos, freq, tSR)
    risk = norm.cdf(thresP, p, p * (1 - p))    # approximation to bootstrap
    return risk

# %% [markdown]
# # Chapter 16. Machine Learning Asset Allocation

# %%
def get_ivp(cov: np.ndarray, **kargs) -> np.ndarray:
    '''
    Computes the inverse variance portfolio.
    
        Parameters:
            cov (np.ndarray): covariance matrix
            
        Returns:
            ivp (np.ndarray): optimal portfolio weights
    '''
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

# %%
def get_cluster_var(cov: np.ndarray, cItems: np.ndarray) -> float:
    '''
    Computes variance per cluster
    
        Parameters:
            cov (np.ndarray): covariance matrix for all items
            cItems (np.ndarray): indexes of cluster items
            
        Returns:
            cVar (float): cluster variance
    '''
    cov_ = cov.loc[cItems, cItems]    # matrix slice
    w_ = get_ivp(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar

# %%
def get_quasi_diag(link: np.ndarray) -> list:
    '''
    Performs Quasi-Diagonalization by sorting clustered items by distance.
    
        Parameters:
            link (np.ndarray): a linkage matrix of size (N−1)x4
        
        Returns:
            lst (list): sorted items list
    '''
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]    # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)    # make space
        df0 = sortIx[sortIx >= numItems]    # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]    # item 1
        df0 = pd.Series(link[j, 1], index=i+1)
        sortIx = sortIx.append(df0)    # item 2
        sortIx = sortIx.sort_index()    # re-sort
        sortIx.index = range(sortIx.shape[0])    # re-index
    lst =  sortIx.tolist()
    return lst

# %%
def get_rec_bipart(cov: np.ndarray, sortIx: list) -> pd.Series:
    '''
    Computes Hierarchical Risk Parity allocation for a given subset of items.
    
        Parameters:
            cov (np.ndarray): covariance matrix
            sortIx (list): sorted items list
    '''
    w = pd.Series([1] * len(sortIx), index=sortIx)
    cItems = [sortIx]    # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[int(j): int(k)] for i in cItems
                  for j, k in ((0, len(i) / 2), (len(i) / 2, len(i))) if len(i) > 1]    # bi-section
        for i in range(0, len(cItems), 2):    # parse in pairs
            cItems0 = cItems[i]    # cluster 1
            cItems1 = cItems[i+1]    # cluster 2
            cVar0 = get_cluster_var(cov, cItems0)
            cVar1 = get_cluster_var(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha    # weight 1
            w[cItems1] *= 1 - alpha    # weight 2
    return w

# %%
def correl_dist(corr: np.ndarray) -> np.ndarray:
    '''
    Calculates a distance matrix based on correlation, where 0<=d[i,j]<=1. This is a proper distance metric.
    
        Parameters:
            corr (np.ndarray): correlation matrix
        
        Returns:
            dist (np.ndarray): distance matrix
    '''
    dist = ((1 - corr) / 2.0) ** 0.5    # distance matrix
    return dist

# %%
def plot_corr_matrix(corr: np.ndarray, labels: list = None, size: tuple = (9, 9)) -> None:
    '''
    Plots heatmap of the correlation matrix.
    
        Parameters:
            corr (np.ndarray): correlation matrix
            labels (list): labels for items
    '''
    fig, ax = plt.subplots(figsize=size)
    if labels is None:
        labels = []
    ax = sns.heatmap(corr)
    ax.set_yticks(np.arange(0.5, corr.shape[0] + 0.5), list(labels))
    ax.set_xticks(np.arange(0.5, corr.shape[0] + 0.5), list(labels))
    plt.show()

# %%
def generate_data(nObs: int, size0: int, size1: int, sigma1: float) -> Tuple[pd.DataFrame, list]:
    '''
    Generates data with correlations.
    
        Parameters:
            nObs (int): number of observations
            size0 (int): number of uncorrelated items
            size1 (int): number of correlated items
            sigma1 (float): std for random noise
            
        Returns:
            x (pd.DataFrame): dataframe with generated data
            cols (list): list with index of correlated items for each of the item in the list
    '''
    #1) generating some uncorrelated data
    np.random.seed(seed=42)
    random.seed(42)
    x = np.random.normal(0, 1, size=(nObs, size0))    # each row is a variable
    #2) creating correlation between the variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma1, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns=range(1, x.shape[1] + 1))
    return x, cols

# %%
def run_simulation() -> None:
    '''
    Runs simulation and performs HRP algorithm.
    '''
    #1) Generate correlated data
    nObs, size0, size1, sigma1 = 10000, 5, 5, 0.25
    x, cols = generate_data(nObs, size0, size1, sigma1)
    print([(j + 1, size0 + i) for i, j in enumerate(cols, 1)])
    cov, corr = x.cov(), x.corr()
    #2) compute and plot correl matrix
    plot_corr_matrix(corr, labels=corr.columns, size=(8, 6.5))
    #3) cluster
    dist = correl_dist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = get_quasi_diag(link)
    sortIx = corr.index[sortIx].tolist()    # recover labels
    df0 = corr.loc[sortIx, sortIx]    # reorder
    plot_corr_matrix(df0, labels=df0.columns, size=(8, 6.5))
    #4) Capital allocation
    hrp = get_rec_bipart(cov, sortIx)
    print(hrp)

# %%
def generate_data_mc(
    nObs: int, sLength: int, size0: int, size1: int, mu0: float, sigma0: float, sigma1F: float
) -> Tuple[np.ndarray, list]:
    '''
    Generates data with two types of random shocks:
    common to various investments and specific to a single investment.
    
    Parameters:
            nObs (int): number of observations
            sLength (int): period length to compute HRP and IVP
            size0 (int): number of uncorrelated items
            size1 (int): number of correlated items
            mu0 (float): mean for generated uncorrelated data
            sigma0 (float): std for random noise
            sigma1F (float): multiplier for correlation noise
            
        Returns:
            x (np.ndarray): matrix with generated data
            cols (list): list with index of correlated items for each of the item in the list
    '''
    #1) generate random uncorrelated data
    x = np.random.normal(mu0, sigma0, size=(nObs, size0))
    #2) create correlation between the variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma0 * sigma1F, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    #3) add common random shock
    point = np.random.randint(sLength, nObs - 1, size=2)
    x[np.ix_(point, [cols[0], size0])] = np.array([[-0.5, -0.5], [2, 2]])
    #4) add specific random shock
    point = np.random.randint(sLength, nObs - 1, size=2)
    x[point, cols[-1]] = np.array([-0.5, 2])
    return x, cols

# %%
def get_hrp(cov: np.ndarray, corr: np.ndarray) -> pd.Series:
    '''
    Constructs a hierarchical portfolio.
    
        Parameters:
            cov (np.ndarray): covariance matrix
            corr (np.ndarray): correlation matrix
            
        Returns:
            hrp (pd.Series): portfolio weight given by HRP method
    '''
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = correl_dist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = get_quasi_diag(link)
    sortIx = corr.index[sortIx].tolist()    # recover labels
    hrp = get_rec_bipart(cov,sortIx)
    return hrp.sort_index()

# %%
def hrp_mc(
    numIters: int = 1e2, nObs: int = 520, size0: int = 5, size1: int = 5, mu0: float = 0,
    sigma0: float = 1e-2, sigma1F: float = 0.25, sLength: int = 260, rebal: int = 22
) -> None:
    '''
    Performs Monte Carlo experiment on HRP method.
    
    Parameters:
        numIters (int): number of Monte Carlo iterations
        nObs (int): number of observations
        size0 (int): number of uncorrelated items
        size1 (int): number of correlated items
        mu0 (float): mean for generated uncorrelated data
        sigma0 (float): std for random noise
        sigma1F (float): multiplier for correlation noise
        sLength (int): period length to compute HRP and IVP
        rebal (int): rebalancing frequency (after how many periods we rebalance portfolios)
    '''
    methods = [get_ivp, get_hrp]
    stats, numIter = {i.__name__: pd.Series() for i in methods}, 0
    pointers = range(sLength, nObs, rebal)
    while numIter < numIters:
        #1) Prepare data for one experiment
        x, cols = generate_data_mc(nObs, sLength, size0, size1, mu0, sigma0, sigma1F)
        r = {i.__name__: pd.Series() for i in methods}
        #2) Compute portfolios in-sample
        for pointer in pointers:
            x_ = x[pointer - sLength: pointer]
            cov_, corr_ = np.cov(x_, rowvar=0), np.corrcoef(x_, rowvar=0)
            #3) Compute performance out-of-sample
            x_ = x[pointer: pointer + rebal]
            for func in methods:
                w_ = func(cov=cov_, corr=corr_)    # callback
                r_ = pd.Series(np.dot(x_, w_))
                r[func.__name__] = r[func.__name__].append(r_)
        #4) Evaluate and store results
        for func in methods:
            r_ = r[func.__name__].reset_index(drop=True)
            p_ = (1 + r_).cumprod()
            stats[func.__name__].loc[numIter] = p_.iloc[-1] - 1
        numIter += 1
    #5) Report results
    stats = pd.DataFrame.from_dict(stats, orient='columns')
    df0, df1 = stats.std(), stats.var()
    print(pd.concat([df0, df1, df1 / df1['get_hrp'] - 1], axis=1))

# %% [markdown]
# # Chapter 17. Structural Breaks

# %%
def get_betas(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    '''
    Carries out the actual regressions.
    
        Parameters:
            y (np.ndarray): y array
            x (np.ndarray): x matrix
            
        Returns:
            bMean (float): estimate of the mean of the beta coefficient
            bVar (float): estimate of the variance of the beta coefficient
    '''
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xxinv = np.linalg.inv(xx)
    bMean = np.dot(xxinv, xy)
    err = y - np.dot(x, bMean)
    bVar = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv
    return bMean, bVar

# %%
def lag_DF(df0: pd.DataFrame, lags: Union[int, list]) -> pd.DataFrame:
    '''
    Applies specified lags to a dataframe.
    
        Parameters:
            df0 (pd.DataFrame): dataframe to which lags are applied
            lags (Union[int, list]): lags
            
        Returns:
            df1 (pd.DataFrame): transformed dataframe
    '''
    df1 = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(lags + 1)
    else:
        lags = [int(lag) for lag in lags]
    for lag in lags:
        df_ = df0.shift(lag).copy(deep=True)
        df_.columns = [str(i) + '_' + str(lag) for i in df_.columns]
        df1 = df1.join(df_, how='outer')
    return df1

# %%
def get_YX(series: pd.Series, constant: str, lags: Union[int, list]) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Prepares numpy objects needed to conduct the recursive texts.
    
        Parameters:
            series (pd.Series): series (containing prices)
            constant (str): the regression's time trend component
                                - 'nc': no time trend, only a constant
                                - 'ct': a constant plus a linear time trend
                                - 'ctt': a constant plus a second-degree polynomial time trend
            lags (Union[int, list]): the number of lags used in the ADF specification
        
        Returns:
            y (np.ndarray): y array
            x (np.ndarray): x matrix
    '''
    series_ = series.diff().dropna()
    x = lag_DF(series_, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0] - 1: -1, 0]    # lagged level
    y = series_.iloc[-x.shape[0]:].values
    if constant != 'nc':
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        if constant[:2] == 'ct':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.append(x, trend, axis=1)
        if constant == 'ctt':
            x = np.append(x, trend ** 2, axis=1)
    return y, x

# %%
def get_bsdaf(logP: pd.Series, minSL: int, constant: str, lags: Union[int, list]) -> dict:
    '''
    Conducts SDAF inner loop.
    
        Parameters:
            logP (pd.Series): series containing log prices
            minSL (int): minimum sample length used by the final regression
            constant (str): the regression's time trend component
                                - 'nc': no time trend, only a constant
                                - 'ct': a constant plus a linear time trend
                                - 'ctt': a constant plus a second-degree polynomial time trend
            lags (Union[int, list]): the number of lags used in the ADF specification
        
        Returns:
            out (dict): dictionary with time and SADF_t estimation
    '''
    y, x = get_YX(logP, constant=constant, lags=lags)
    startPoints, bsadf, allADF = range(0, y.shape[0] + lags - minSL + 1), None, []
    for start in startPoints:
        y_, x_ = y[start:], x[start:]
        bMean_, bStd_ = get_betas(y_, x_)
        bMean_, bStd_ = bMean_[0, 0], bStd_[0, 0] ** 0.5
        allADF.append(bMean_ / bStd_)
        if allADF[-1] > bsadf:
            bsadf = allADF[-1]
    out = {'Time': logP.index[-1], 'gsadf': bsadf}
    return out

# %% [markdown]
# # Chapter 18. Entropy Features

# %%
def pmf1(msg: Any, w: int) -> dict:
    '''
    Computes the probability mass function for a one-dim random variable (len(msg) - w occurences).
    
        Parameters:
            msg (Any): sequence with observations (usually a string)
            w (int): word length used for pmf estimation
            
        Returns:
            pmf (dict): dictionary with words as keys and their estimated probabilities as values.
    '''
    lib = {}
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    for i in range(w, len(msg)):
        msg_ = msg[i - w: i]
        if msg_ not in lib:
            lib[msg_] = [i - w]
        else:
            lib[msg_] = lib[msg_] + [i - w]
    length = float(len(msg) - w)
    pmf = {i: len(lib[i]) / length for i in lib}
    return pmf

# %%
def plug_in(msg: Any, w: int) -> Tuple[float, dict]:
    '''
    Computes the maximum likelihood estimate for the entropy rate.

        Parameters:
            msg (Any): sequence with observations (usually a string)
            w (int): word length used for pmf estimation

        Returns:
            out (float): entropy estimate
            pmf (dict): dictionary with words as keys and their estimated probabilities as values.
    '''
    pmf = pmf1(msg, w)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / w
    return out, pmf

# %%
def lempel_ziv_lib(msg: str) -> list:
    '''
    Implements the LZ algorithm to construct library.
    
        Parameters:
            msg (str): sequence with observations
            
        Returns:
            lib (list): list containing unique words
    '''
    i, lib = 1, [msg[0]]
    while i < len(msg):
        for j in range(i, len(msg)):
            msg_ = msg[i: j + 1]
            if msg_ not in lib:
                lib.append(msg_)
                break
        i = j + 1
    return lib

# %%
def match_length(msg: str, i: int, n: int) -> Tuple[int, str]:
    '''
    Computes the length of the longest match.
    
        Parameters:
            msg (str): sequence with observations
            i (int): position before which we look for a match
            n (int): size of the window for searching for a match
            
        Returns:
            len(subS) + 1 (int): length of the match + 1
            subS (str): matched substring
    '''
    subS = ''
    for l in range(n):
        msg1 = msg[i: i + 1 + l]
        for j in range(i - n, i):
            msg0 = msg[j: j + 1 + l]
            if msg1 == msg0:
                subS = msg1
                break
    return len(subS) + 1, subS

# %%
def konto(msg: Any, window: Optional[int] = None) -> dict:
    '''
    Kontoyiannis' LZ entropy estimate, 2013 version (centered window). Inverse of the avg length of the shortest
    non redundant substring. If non redundant substrings are short, the text is highly entropic.
    window=None for expanding window , in which case len(msg)%2=0.
    If the end of the message is more relevant, try conto(msg[::-1]).
    
        Parameters:
            msg (Any): sequence with observations (usually a string)
            window (Optional[int]): winodw size for constant window
            
        Returns:
            out (dict): dictionary with results
    '''
    out = {'num': 0, 'sum': 0, 'subS': []}
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    if window is None:
        points = range(1, len(msg) // 2 + 1)
    else:
        window = min(window, len(msg) // 2)
        points = range(window, len(msg) - window + 1)
    for i in points:
        if window is None:
            l, msg_ = match_length(msg, i, i)
            out['sum'] += np.log2(i + 1) / l    # to avoid Doeblin condition
        else:
            l, msg_ = match_length(msg, i, window)
            out['sum'] += np.log2(window + 1) / l    # to avoid Doeblin condition
        out['subS'].append(msg_)
        out['num'] += 1
    out['h'] = out['sum'] / out['num']
    out['r'] = 1 - out['h'] / np.log2(len(msg))    # redundancy, 0 <= r <= 1
    return out


