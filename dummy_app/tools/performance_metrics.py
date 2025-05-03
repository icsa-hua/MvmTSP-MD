import os 
import time 
import psutil
import tracemalloc 
from dummy_app.tools.logger import logger 


class Metrics: 

    def __init__(self, verbose:bool=False): 
        self.start_time = None 
        self.end_time = None 
        self.memory_usage = None 
        self.variables = None 
        self.constraints = None 


    def start_tracemalloc(self): 
        tracemalloc.start()

    def start_performance_timer(self):
        self.start_time = time.time() 

    def end_performance_timer(self):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        self.memory_usage = process.memory_info().rss / (1024 ** 2)

    def display_memory_usage(self): 
        snapshot = tracemalloc.take_snapshot() 
        top_stats = snapshot.statistics('lineno') 
        
        logger.debug("Memory allocation snapshot: ")
        logger.debug(f"The top memory-consuming variable: {top_stats[0]}")
        logger.debug(f"Total allocated memory: {top_stats[0].size / (1024 ** 2)} MB")   

        if self.verbose: 
            logger.info(f"Total allocated memory: {top_stats[0].size / (1024 ** 2)} MB")

        
    

    
    
    