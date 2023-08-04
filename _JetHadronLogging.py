print("Setting up logs for JetHadronAnalysis...")


import logging

debug_logger = logging.getLogger('debug')
error_logger = logging.getLogger('error')
info_logger = logging.getLogger('info')
unfolding_logger = logging.getLogger('unfolding')
d_fh = logging.FileHandler(filename='jhDEBUG.log', mode='w', encoding='utf-8',)
e_fh = logging.FileHandler(filename='jhERROR.log', mode='w', encoding='utf-8',)
i_fh = logging.FileHandler(filename='jhINFO.log', mode='w', encoding='utf-8',)
d_fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
e_fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
i_fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
debug_logger.addHandler(d_fh)
error_logger.addHandler(e_fh)
info_logger.addHandler(i_fh)
unfolding_logger.addHandler(logging.FileHandler('unfolding.log', mode='w'))
debug_logger.setLevel(logging.DEBUG)
error_logger.setLevel(logging.ERROR)
info_logger.setLevel(logging.INFO)
unfolding_logger.setLevel(logging.INFO)
debug_logger.info("Debug logger initialized.")
error_logger.info("Error logger initialized.")
info_logger.info("Info logger initialized.")
# let's register a new file for the unfolding logger

def log_function_call(description, logging_level=logging.DEBUG):
    """
    Logs the function call with the given description and logging level.
    """

    def function_wrapper(function):
        def method_wrapper(self, *args, **kwargs):
            if logging_level == logging.DEBUG:
                logger = debug_logger
            elif logging_level == logging.INFO:
                logger = info_logger
            elif logging_level == logging.ERROR:
                logger = error_logger
            else:
                raise ValueError(f"Unknown logging level {logging_level}")
            logger.log(level=logging_level, msg=f"{function.__name__} in {self.__class__.__name__}:\n\t{description}")
            return function(self, *args, **kwargs)

        return method_wrapper

    return function_wrapper