import logging
import datetime
import pytz

class BeijingTimeFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, time_offset=datetime.timedelta(0)):
        super().__init__(fmt, datefmt)
        self.time_offset = time_offset

    def formatTime(self, record, datefmt=None):
        # Convert the record's created time to UTC, then to Beijing time with an optional offset
        utc_time = datetime.datetime.utcfromtimestamp(record.created)
        beijing_tz = pytz.timezone('Asia/Shanghai')
        beijing_time = utc_time.astimezone(beijing_tz) + self.time_offset
        return beijing_time.strftime('%Y-%m-%d %H:%M:%S')

def get_logger(save_dir):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(save_dir + 'log.txt', mode='w')
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    time_offset = datetime.timedelta(seconds=-198)
    formatter = BeijingTimeFormatter('%(asctime)s-%(levelname)s: %(message)s', time_offset=time_offset)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger