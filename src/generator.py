"""
[TR] Test amaçlı "kirli" (hatalı) sensör verisi ve her satır için hata etiketi üretir.
[EN] Generates "dirty" (faulty) sensor data plus a fault label for every row.
"""
import numpy as np
import pandas as pd

OK = "ok"
SPIKE = "spike"    # [TR] Fiziksel olarak imkansız sıçrama  [EN] Physically impossible jump
FROZEN = "frozen"  # [TR] Donmuş (takılı kalmış) sensör     [EN] Stuck / frozen sensor


class FaultyDataGenerator:
    def __init__(self, seed=10):
        self.seed = seed

    def generate_data(self, n=500):
        """[TR] (veri, etiketler) döndürür. Etiketler: 'ok', 'spike', 'frozen'.
           [EN] Returns (data, labels). Labels: 'ok', 'spike', 'frozen'."""
        rng = np.random.RandomState(self.seed)
        # [TR] Normal temiz veri  [EN] Clean baseline signal
        temp = 70 + rng.normal(0, 1, n)
        labels = np.full(n, OK, dtype=object)

        # [TR] Kasıtlı hatalar  [EN] Injected faults
        temp[100:110] = 180
        labels[100:110] = SPIKE
        temp[300:320] = 70  # [TR] Ortalamaya eşit -> Z-skoru yakalayamaz  [EN] Equals the mean -> invisible to Z-score
        labels[300:320] = FROZEN

        df = pd.DataFrame({"temperature": temp})
        return df, pd.Series(labels, index=df.index, name="fault")
