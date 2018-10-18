"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from __future__ import division
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume, CustomFactor, Returns
from quantopian.pipeline import CustomFilter
import numpy as np
import pandas as pd
from scipy import stats
 
 

class SharpeRatio(CustomFactor):
    # inputs = [returns]
    window_safe = True

    def compute(self, today, assets, out, yr_returns):
        out[:] = np.nanmean(yr_returns, axis=0) / np.nanstd(yr_returns, axis=0)

 
 
 

class SecurityInList(CustomFactor):
    inputs = []
    window_length = 1
    securities = []

    def compute(self, today, assets, out):
        out[:] = np.in1d(assets, self.securities)


def initialize(context):
    """
    Called once at the start of the algorithm.
    """

    context.counter = 0
    set_benchmark(sid(8554))
    # Rebalance at the #end of every month, 1 hour after market open.
    schedule_function(my_assign_weights_p98, date_rules.month_end(), time_rules.market_open())
    schedule_function(my_rebalance, date_rules.month_end(), time_rules.market_open(hours=1))

    # Record tracking variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())

    # Create our dynamic stock selector.
    context.return_period = 252
    # SPY
    context.mom1 = mom1 = sid(8554)
    # VEU
    context.mom2 = mom2 = sid(33486)
    # SHV
    context.tbill = tbill = sid(33154)
    # BND
    context.agg = agg = sid(33652)
    # QQQ - NASDAQ
    context.tech = tech = sid(39214)

    context.stock_leverage = 1
    sec_list = [tech, mom1, mom2, tbill, agg]
    attach_pipeline(make_pipeline(sec_list, context), 'my_pipeline')

 
set_commission(commission.PerShare(cost=0, min_trade_cost=0))


# Momentum ETFs

def make_pipeline(sec_list, context):
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """

    # Return Factors
    mask = SecurityInList()
    mask.securities = sec_list
    mask = mask.eq(1)
    yr_returns = Returns(window_length=context.return_period, mask=mask)
    sharpe = SharpeRatio(inputs=[yr_returns], window_length=context.return_period, mask=mask)
    pipe = Pipeline(
        screen=mask,
        columns={
            'yr_returns': yr_returns
        }
    )
    return pipe

 
 
 
 

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = pipeline_output('my_pipeline')

 

def my_assign_weights_p98(context, data):
    context.weights = pd.Series(index=context.output.index)
    returns = context.output['yr_returns']

 
if returns[context.mom1] < returns[context.tbill]:
    context.weights[context.agg] = 1

elif returns[context.mom1] > returns[context.mom2]:
    context.weights[context.mom1] = context.stock_leverage

else:
    context.weights[context.mom2] = context.stock_leverage
 
context.weights.fillna(0, inplace=True)


def my_assign_weights(context, data):
    context.weights = pd.Series(index=context.output.index)  #
    returns = context.output['yr_returns']

    if returns[context.tech] > returns[context.mom1]:
        if returns[context.tech] > returns[context.tbill]:
            context.weights[context.tech] = context.stock_leverage
        else:
            context.weights[context.agg] = 1

    elif returns[context.mom1] > returns[context.mom2]:
        if returns[context.mom1] > returns[context.tbill]:
            context.weights[context.mom1] = context.stock_leverage
        else:
            context.weights[context.agg] = 1

    else:
        if returns[context.mom2] > returns[context.tbill]:
            context.weights[context.mom2] = context.stock_leverage
        else:
            context.weights[context.agg] = 1

    context.weights.fillna(0, inplace=True)


def my_rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    freq_month = 1
    context.counter += 1
    if context.counter == freq_month:
        for stock, weight in context.weights.iteritems():
            context.counter = 0
            if data.can_trade(stock):
                order_target_percent(stock, weight)


def my_record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(leverage=context.account.leverage)


def handle_data(context, data):
    """
    Called every minute.
    """
    pass