from utils.logging.logger import Logger


class MultiLogger(Logger):
    """
    A logger that logs to multiple loggers.
    """

    def __init__(self, *loggers: Logger):
        config = loggers[0].config if len(loggers) > 0 else {}
        super().__init__(config)
        for logger in loggers:
            if logger.config != config:
                raise ValueError("All loggers must have the same config.")
        self.loggers = loggers

    def log_config(self):
        for logger in self.loggers:
            logger.log_config()

    def log(self, data: dict[str, float | list[float]]):
        for logger in self.loggers:
            logger.log(data)

    def finish(self):
        for logger in self.loggers:
            logger.finish()
