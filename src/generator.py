"""
[TR] Test amaçlı "kirli" (hatalı/kaymış) sensör verileri üretir.
[EN] Generates "dirty" (faulty/drifted) sensor data for testing.
"""
import numpy as np
import pandas as pd

class FaultyDataGenerator:
    def generate_data(self, n=500):
        np.random.seed(10)
        # [TR] Normal temiz veri
        temp = 70 + np.random.normal(0, 1, n)
        
        # [TR] Kasıtlı hatalar ekleyelim: Sensör donması veya aşırı ısınma
        # [EN] Inject intentional faults: Sensor freeze or overheating
        temp[100:110] = 180  # [TR] Fiziksel olarak imkansız bir sıçrama
        temp[300:320] = 70   # [TR] Donmuş sensör simülasyonu
        
        return pd.DataFrame({'temperature': temp})