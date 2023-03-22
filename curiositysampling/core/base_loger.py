import logging

# from https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
# class CustomFormatter(logging.Formatter):
#
#    grey = "\x1b[38;20m"
#    yellow = "\x1b[33;20m"
#    red = "\x1b[31;20m"
#    bold_red = "\x1b[31;1m"
#    reset = "\x1b[0m"
#    format_max = " %(name)s - %(asctime)s :: %(levelname)s - %(message)s :: (%(filename)s:%(lineno)d)"
#    format_min = " %(name)s - %(asctime)s :: %(levelname)s - %(message)s"
#
#    FORMATS = {
#        logging.DEBUG: grey + format_max + reset,
#        logging.INFO: grey + format_min + reset,
#        logging.WARNING: yellow + format_max + reset,
#        logging.ERROR: red + format_max + reset,
#        logging.CRITICAL: bold_red + format_max + reset,
#    }
#
#    def format(self, record):
#        log_fmt = self.FORMATS.get(record.levelno)
#        formatter = logging.Formatter(log_fmt)
#        return formatter.format(record)
#
#
# logger = logging.getLogger("Curiosity Sampling")

# formatter = CustomFormatter()
# ch = logging.StreamHandler()
# ch.setFormatter(formatter)
# logger.addHandler(ch)
logger = logging
logger.basicConfig(
    format=" Curiosity Sampling - %(asctime)s :: %(levelname)s - %(message)s :: (%(filename)s:%(lineno)d)",
    level=logging.INFO,
)
