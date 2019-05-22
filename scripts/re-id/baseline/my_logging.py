#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samon
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: my_logging.py
# @Date: 19-5-18 下午5:21
# @Descr:
"""
logging配置
"""

import logging.config
import os
import time


__all__ = ['init_logging']

# 定义三种日志输出格式
full_format = '[%(asctime)s][%(threadName)s:%(thread)d][task_id:%(name)s][%(filename)s:%(lineno)d]' \
                  '[%(levelname)s][%(message)s]'  # 其中name为getlogger指定的名字
standard_format = '[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d]%(message)s'
simple_format = '[%(levelname)s][%(asctime)s] %(message)s'
tag_format = '[%(tag)s][%(asctime)s] %(message)s'


def init_logging(logfile_path=None):
    logfile_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(logfile_dir):
        os.mkdir(logfile_dir)

    if logfile_path is None:
        # log文件的全路径
        date_time = time.strftime("%m%d-%H%M")
        logfile_path = os.path.join(logfile_dir, '{}.log'.format(date_time))

    # log备份个数
    backup_count = 5
    # 单个日志最大容量
    max_bytes = 1024 * 1024 * 10  # 日志大小 10M

    # log配置字典
    LOGGING_DIC = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': standard_format
            },
            'simple': {
                'format': simple_format,
                'datefmt': '%m-%d %H:%M:%S',
            },
            'tag_format': {
                'format': tag_format,
                'datefmt': '%m-%d %H:%M:%S',
            },
        },
        'filters': {},
        'handlers': {
            # 定义打印到终端的日志
            'console_simple': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',  # 打印到屏幕
                'formatter': 'simple'
            },
            # 定义打印到文件的日志
            'file_standard': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',  # 保存到文件
                'formatter': 'standard',
                'filename': logfile_path,  # 日志文件
                'maxBytes': max_bytes,  # 日志大小 10M
                'backupCount': backup_count,
                'encoding': 'utf-8',  # 日志文件的编码，防止中文log乱码
            },
            # 定义打印到终端的日志
            'console_tag': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'tag_format',
            },
            # 定义打印到终端的日志
            'file_tag': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'tag_format',
                'filename': logfile_path,  # 日志文件
                'maxBytes': max_bytes,  # 日志大小 10M
                'backupCount': backup_count,
                'encoding': 'utf-8',  # 日志文件的编码，防止中文log乱码
            },
        },
        'loggers': {
            # 可以logging.getLogger(__name__)拿到的logger配置
            "": {
                'handlers': ['file_standard', 'console_simple'],  # 既写入文件又打印到屏幕
                'level': 'DEBUG',
                'propagate': False,  # 不向上（更高level的logger）传递
            },
            "tag": {
                'handlers': ['console_tag', 'file_tag'],  # 既写入文件又打印到屏幕
                'level': 'DEBUG',
                'propagate': False,  # 不向上（更高level的logger）传递
            },
        },
    }

    logging.config.dictConfig(LOGGING_DIC)  # 导入logging配置


if __name__ == '__main__':
    init_logging()
    logging.info("test log")
    logger = logging.getLogger()
    logger.info('default log')

    logger2 = logging.getLogger(name="tag")
    logger2.info(msg='It works!', extra={'tag': '10.100.1.98'})
