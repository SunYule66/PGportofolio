#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import path

def get_database_path(database_file=None):
    """
    获取数据库路径
    :param database_file: 数据库文件名，如果为None则使用默认的Data.db
    :return: 数据库文件的完整路径
    """
    if database_file is None:
        database_file = "Data.db"
    base_path = path.realpath(__file__).\
        replace('pgportfolio/constants.pyc','/database/').\
        replace("pgportfolio\\constants.pyc","database\\").\
        replace('pgportfolio/constants.py','/database/').\
        replace("pgportfolio\\constants.py","database\\")
    return path.join(base_path, database_file)

# 默认数据库路径（保持向后兼容）
DATABASE_DIR = get_database_path()
CONFIG_FILE_DIR = 'net_config.json'
LAMBDA = 1e-4  # lambda in loss function 5 in training
   # About time
NOW = 0
FIVE_MINUTES = 60 * 5
FIFTEEN_MINUTES = FIVE_MINUTES * 3
HALF_HOUR = FIFTEEN_MINUTES * 2
HOUR = HALF_HOUR * 2
TWO_HOUR = HOUR * 2
FOUR_HOUR = HOUR * 4
DAY = HOUR * 24
YEAR = DAY * 365
   # trading table name
TABLE_NAME = 'test'

