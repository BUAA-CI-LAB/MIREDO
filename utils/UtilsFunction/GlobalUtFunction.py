# this file is prepared for project 511
# Created by iboxl

import logging
import os
import sys
import io

class _levelFilter(logging.Filter):
        def __init__(self, level):
            self.level = level
        def filter(self, record):
            return record.levelno == self.level

class CustomFormatterFile(logging.Formatter):
    """自定义格式器，根据不同的日志级别输出不同的格式。"""
    def __init__(self, default_fmt="%(message)s"):
        super().__init__(default_fmt)

    def format(self, record):
        # 根据日志级别动态设置格式
        if record.levelno == logging.DEBUG:
            self._style._fmt = "🟠Debug: %(message)s"
        elif record.levelno == logging.WARNING:
            self._style._fmt = "🔴Warning: %(message)s"
        elif record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        elif record.levelno == logging.ERROR:
            self._style._fmt = "🔴Error: %(message)s"
        elif record.levelno == logging.CRITICAL:
            self._style._fmt = "🔵Critical: %(message)s"
        else:
            self._style._fmt = "%(message)s"  # 默认格式

        # 使用设置的格式生成格式化后的记录
        result = super().format(record)
        
        # 恢复默认格式
        self._style._fmt = "%(message)s"

        return result
    
class CustomFormatterConsole(logging.Formatter):
    """自定义格式器，根据不同的日志级别输出不同的格式。"""
    RED = "\x1b[31m"        # 红色
    GREEN = "\x1b[32m"      # 绿色
    YELLOW = "\x1b[33m"     # 黄色
    BLUE = "\x1b[34m"       # 蓝色
    MAGENTA = "\x1b[35m"    # 洋红色
    CYAN = "\x1b[36m"       # 青色
    RESET = "\x1b[0m"       # 重置颜色
    def __init__(self, default_fmt="%(message)s"):
        super().__init__(default_fmt)

    def format(self, record):
        # 根据日志级别动态设置格式
        if record.levelno == logging.DEBUG:
            self._style._fmt = self.YELLOW+"Debug: "+self.RESET+"%(message)s"
        elif record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.RED+"Warning: "+self.RESET+"%(message)s"
        elif record.levelno == logging.ERROR:
            self._style._fmt = self.RED+"Error: "+self.RESET+"%(message)s"
        elif record.levelno == logging.CRITICAL:
            self._style._fmt = self.BLUE+"Critical: "+self.RESET+"%(message)s"
        else:
            self._style._fmt = "%(message)s"  # 默认格式

        # 使用设置的格式生成格式化后的记录
        result = super().format(record)
        
        # 恢复默认格式
        self._style._fmt = "%(message)s"

        return result

class _Logger():    
    def __init__(self, name="511", file="511.log", level=logging.INFO, levelFilter=None):
        self.original_stdout = sys.stdout
        self.file = file
        self.levelFilter = levelFilter
        # 创建一个日志对象
        self.logger = logging.getLogger(name)
        self.logger.setcfg = self.setcfg
        self.logger.recover_stdout = self.recover_stdout
        self.logger.propagate = False                      # Avoid the influence of third-party lib

        self.logger.changeFile = self.changeFile
        self._file_handler = None
        self._console_handler = None

    def get_Logger(self):
        return self.logger
    
    def recover_stdout(self):
        sys.stdout = self.original_stdout

    def _remove_all_handlers(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        self._file_handler = None
        self._console_handler = None
    
    def changeFile(self, new_file: str, mode: str = "a") -> None:
        """
        动态替换 FileHandler 并写入新的日志文件。
        - new_file: 新文件路径
        - mode: 打开模式，默认 'a' 追加；若想每次覆盖写用 'w'
        """
        # 若目录不存在则自动创建
        dirname = os.path.dirname(new_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # 1) 移除并关闭旧 FileHandler
        if self._file_handler is not None:
            self.logger.removeHandler(self._file_handler)
            self._file_handler.close()

        # 2) 新建 FileHandler
        fh = logging.FileHandler(new_file, mode=mode)
        fh.setFormatter(CustomFormatterFile())
        if self.levelFilter is not None:
            fh.addFilter(_levelFilter(self.levelFilter))

        self.logger.addHandler(fh)
        self._file_handler = fh          # 更新内部引用
            
    
    def setcfg(self, setcritical, setDebug, STD, file, nofile):      
         
        if STD:
            sys.stdout = io.StringIO()    

        self._remove_all_handlers()

        root_logger = logging.getLogger()
        # 移除所有已经存在的处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建一个日志处理器，将日志写入日志文件
        if not nofile:
            self.changeFile(file, mode="w")   # 直接复用 change_file
        else:
            # 若用户要求 nofile=True，则确保关闭之前的文件句柄
            if self._file_handler is not None:
                self.logger.removeHandler(self._file_handler)
                self._file_handler.close()
                self._file_handler = None

        # 创建一个日志处理器，将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatterConsole())
        if self.levelFilter is not None:
            console_handler.addFilter(_levelFilter(self.levelFilter))
        self.logger.addHandler(console_handler)
        self._console_handler = console_handler
        
        if setcritical:
            self.logger.setLevel(logging.CRITICAL)
        elif setDebug:                                # 设置日志级别
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # for handler in self.logger.handlers[:]:
        #     self.logger.removeHandler(handler)

        # if nofile is False:
        #     self.logger.addHandler(file_handler)  # 文件输出
        # self.logger.addHandler(console_handler)  # 控制台输出
