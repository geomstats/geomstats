import logging


logging.basicConfig(format='%(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
if loggers and loggers[0].name.startswith('nose2'):
    logging.getLogger().setLevel(logging.WARNING)
