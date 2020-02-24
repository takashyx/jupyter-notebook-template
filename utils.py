# -*- coding: utf-8 -*-

import datetime
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pprint as pp
import numpy as np
import traceback
import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)


# TODO クラス化

def binary_encode(df, columns, drop_invariant=True):
    encoder = ce.BinaryEncoder(
        cols=columns,
        drop_invariant=drop_invariant)
    return encoder.fit_transform(df)


def int_str_to_int(int_str, nan_value=-1):
    try:
        if isinstance(int_str, str):
            ret = int_str.replace(",", "").replace(' ', '')
            if ret == '':
                return nan_value
            else:
                return int(ret)

        elif not isinstance(int_str, str) and math.isnan(int_str):
            return nan_value

    except BaseException:
        import traceback
        traceback.print_exc()
        print("int_str_to_int(): error while parsing:")
        print(int_str)


def float_str_to_float(float_str, nan_value=0):
    try:
        if not isinstance(float_str, str) and math.isnan(float_str):
            return nan_value
        elif isinstance(float_str, float):
            return float(float_str)
        else:
            return float(float_str.replace(",", ""))
    except BaseException:
        import traceback
        traceback.print_exc()
        print("float_str_to_float(): error while parsing:")
        print(float_str)


# unique
def get_unique_values_of_column(df, column):
    return list(df[column].unique())

# remove columns with no variety


def remove_columns_with_no_variety(df):
    columns = df.columns
    for col in columns:
        if len(get_unique_values_of_column(df, col)) == 1:
            df.drop(columns=[col], inplace=True)

    return df


# 1つでもNaNが含まれるカラムを抽出
def check_columns_with_null(df, columns=[]):
    if columns == []:
        print("columns with null value(True has NaN column):")
        print(df.isnull().any(axis=0))
        print('-----')
    else:
        print(f"columns {columns} null value(True has NaN column):")
        print(df[columns].isnull().any(axis=0))
        print('-----')

    print("null containing lines(first 10):")
    print(df[df[columns].isnull().any(axis=1)].head(10))
    print('-----')


# 対象columnにあるNaNをvalに置き換える
def replace_nan_in_columns(df, columns, val):
    for col in columns:
        df.fillna(value={col: val}, inplace=True)
    return df


# 相関グラフ
def draw_corr_graph(df, output_file='---'):
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    sns.heatmap(df.dropna().corr(), vmax=1, vmin=-1,
                center=0, cmap=sns.color_palette("coolwarm", 128))
    if output_file != '---':
        fig.savefig(output_file, bbox_inches='tight')


def time_to_cyclic_sin_cos(time_str, nan_value=0):
    # 10:12 -> sin cos in a day
    if time_str == 0:
        return nan_value, nan_value
    if not isinstance(time_str, str) and math.isnan(time_str):
        return nan_value, nan_value

    if len(time_str) > 2:
        parts = time_str.split(':')
        if len(parts) == 2:
            m = int(parts[0]) * 60 + int(parts[1])
            minutes_in_a_day = 24 * 60
            sin = np.sin(2 * np.pi * m / minutes_in_a_day)
            cos = np.cos(2 * np.pi * m / minutes_in_a_day)
            return sin, cos

        else:
            return nan_value, nan_value
    else:
        return nan_value, nan_value


def datetime_to_cyclic_year(data, nan_value=0):
    # leap year is not considered

    #    if not isinstance(data, datetime):
    #       return nan_value, nan_value

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    d = 0
    try:
        for month in range(data.month):
            d += days_in_month[month]
        d += data.day
        day_in_a_year = sum(days_in_month)
        sin = np.sin(2 * np.pi * d / day_in_a_year)
        cos = np.cos(2 * np.pi * d / day_in_a_year)
        return sin, cos
    except BaseException:
        pp.pprint(data)
        print('-----traceback-----')
        traceback.print_exec()


def datetime_to_cyclic_month(data, nan_value=0):
    # leap year is not considered

    # if not isinstance(data, datetime):
    # return nan_value, nan_value

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_in_the_month = days_in_month[data.month - 1]
    sin = np.sin(2 * np.pi * data.day / day_in_the_month)
    cos = np.cos(2 * np.pi * data.day / day_in_the_month)
    return sin, cos


def datetime_to_cyclic_week(data, nan_value=0):

    # if not isinstance(data, datetime):
    #   return nan_value, nan_value

    day_in_the_week = 7
    sin = np.sin(2 * np.pi * data.weekday() / day_in_the_week)
    cos = np.cos(2 * np.pi * data.weekday() / day_in_the_week)
    return sin, cos


def datetime_cyclic(df, columns, drop_orig_columns=False):
    for col in columns:
        temp = df[col].map(datetime_to_cyclic_year)
        df[col + '_yesr_sin'] = temp.map(lambda x: x[0])
        df[col + '_yesr_cos'] = temp.map(lambda x: x[1])
        temp = df[col].map(datetime_to_cyclic_month)
        df[col + '_month_sin'] = temp.map(lambda x: x[0])
        df[col + '_month_cos'] = temp.map(lambda x: x[1])
        temp = df[col].map(datetime_to_cyclic_week)
        df[col + '_week_sin'] = temp.map(lambda x: x[0])
        df[col + '_week_cos'] = temp.map(lambda x: x[1])
    if drop_orig_columns:
        df.drop(columns=columns, inplace=True)
    return df


def datetime_parser(ts):
    try:
        # input text format: yyyy?mm?dd?hh?mm
        dt = datetime.datetime(int(ts[0:4]), int(ts[5:7]), int(
            ts[8:10]), int(ts[11:13]), int(ts[14:16]), int(0))
    except BaseException:
        print(f"datetime_parser error!  ts: {ts}\n")
        print("-----traceback:-----")
        traceback.print_exc()

    return dt


def datetime_parse(df, columns, inplace=False):
    if inplace:
        for col in columns:
            df[col] = df[col].apply(datetime_parser)
    else:
        for col in columns:
            df[col + '_datetime'] = df[col].apply(datetime_parser)
    return df


def create_data_for_date_range(data, start, end, freq='1D'):
    # start = '2020/01/01'
    # start = '2020/01/05'
    # frep = '1D'
    # data = (1,2,3,4,5)
    return pd.DataFrame({'data': data}, index=pd.date_range(
        start=start, end=end, freq=freq)


if __name__ == "__main__":
    print("remember pd.DataFrame.asfreq/resample/rolling")
    print("remember pd.date_range")
    pass

# todo テスト書く
