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
    def __init__(self, default_fmt="%(message)s"):
        super().__init__(default_fmt)

    def format(self, record):
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
            self._style._fmt = "%(message)s"

        result = super().format(record)

        self._style._fmt = "%(message)s"

        return result

class CustomFormatterConsole(logging.Formatter):
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    RESET = "\x1b[0m"
    def __init__(self, default_fmt="%(message)s"):
        super().__init__(default_fmt)

    def format(self, record):
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
            self._style._fmt = "%(message)s"

        result = super().format(record)

        self._style._fmt = "%(message)s"

        return result

class _Logger():
    def __init__(self, name="511", file="511.log", level=logging.INFO, levelFilter=None):
        self.original_stdout = sys.stdout
        self.file = file
        self.levelFilter = levelFilter
        self.logger = logging.getLogger(name)
        self.logger.setcfg = self.setcfg
        self.logger.recover_stdout = self.recover_stdout
        self.logger.propagate = False

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
        dirname = os.path.dirname(new_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        if self._file_handler is not None:
            self.logger.removeHandler(self._file_handler)
            self._file_handler.close()

        fh = logging.FileHandler(new_file, mode=mode)
        fh.setFormatter(CustomFormatterFile())
        if self.levelFilter is not None:
            fh.addFilter(_levelFilter(self.levelFilter))

        self.logger.addHandler(fh)
        self._file_handler = fh


    def setcfg(self, setcritical, setDebug, STD, file, nofile):

        if STD:
            sys.stdout = io.StringIO()

        self._remove_all_handlers()

        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        if not nofile:
            self.changeFile(file, mode="w")
        else:
            if self._file_handler is not None:
                self.logger.removeHandler(self._file_handler)
                self._file_handler.close()
                self._file_handler = None

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatterConsole())
        if self.levelFilter is not None:
            console_handler.addFilter(_levelFilter(self.levelFilter))
        self.logger.addHandler(console_handler)
        self._console_handler = console_handler

        if setcritical:
            self.logger.setLevel(logging.CRITICAL)
        elif setDebug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
