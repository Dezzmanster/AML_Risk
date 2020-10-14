import time
import logging
import os
from glob import glob
import datetime as dt
from pandas.core.indexes.range import RangeIndex
import pandas as pd

is_start = None
logger = logging.getLogger(__name__)


def timeit(method):
    """
    Decorator for counting execution time and for logging it.
    :param method: decorating method
    :return: wrapped method
    """
    def timed(*args, **kw):
        global is_start

        is_start = True
        logging.info("Start {}.".format(method.__qualname__))

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        logging.info("End {}. Time: {:0.2f} sec.".format(method.__qualname__, end_time - start_time))
        is_start = False

        return result

    return timed


def check_path(path):
    return os.path.exists(path)


def check_dir(path):
    return os.path.isdir(path)


def check_csv(path):
    """
    Check if csv files are existed in particular dir.
    :param path: path to dir, where csv files
    :return: True if csv files are contained, False - vice versa
    """
    is_csv = False
    list_files_in_folder = sorted(glob(os.path.join(path, '*.csv')))
    if len(list_files_in_folder) > 0:
        is_csv = True
    return is_csv


def check_csv_files(path, files):
    """
    Compare csv files in dir and csv files for Generator class.
    :param path: path to dir, where csv files
    :param files: needed csv files for Generator class
    :return: True if csv needed csv files in dir, False - vice versa and List of csv files, that are not founded in dir
    """
    list_files_in_folder = sorted(glob(os.path.join(path, '*.csv')))
    list_needed_files = [os.path.join(path, file) for file, _ in files.items()]

    list_not_founded_files = list()
    for needed_file in list_needed_files:
        if needed_file not in list_files_in_folder:
            list_not_founded_files.append(needed_file)

    all_correct = False
    returned_list_files = list()
    if len(list_not_founded_files) == 0:
        all_correct = True
    else:
        returned_list_files = list_not_founded_files

    return all_correct, returned_list_files


def check_col_in_df(df, col):
    """
    Check if DataFrame object contains particular column.
    :param df: DataFrame object
    :param col: column for check
    :return: True if column in DataFrame object, False vice versa
    """
    col_in_df = False
    if col:
        if col in df.columns:
            col_in_df = True
    elif col is False:
        col_in_df = True
    return col_in_df


@timeit
def save_dataframe_to_csv(df, path, sep):
    """
    Reset index if DataFrame contains a custom index and saving DataFrame to csv file.
    :param df: DataFrame object
    :param path: path to dir for csv file
    :param sep: separator
    """
    if not isinstance(df.index, RangeIndex):
        df.reset_index().to_csv(os.path.join(path), sep=sep, index=False)
    else:
        df.to_csv(os.path.join(path), sep=sep, index=False)


def date_parser(date_in_str):
    """
    Parse date from str to datetime.date format. Format
    :param date_in_str: date in str format (%Y-%m-%d)
    """
    return pd.to_datetime(date_in_str, format='%Y-%m-%d', errors='coerce')




