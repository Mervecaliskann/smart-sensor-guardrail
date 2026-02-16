"""
[TR] Sensör verilerinin kalitesini ve istatistiksel geçerliliğini denetleyen sınıf.
[EN] Class that validates the quality and statistical validity of sensor data.
"""
import pandas as pd
import numpy as np

class SensorValidator:
    def __init__(self, thresholds):
        # [TR] Her sensör için kabul edilebilir alt ve üst limitler
        # [EN] Lower and upper acceptable limits for each sensor
        self.thresholds = thresholds 

    def check_ranges(self, df):
        """[TR] Verinin fiziksel sınırlar içinde olup olmadığını kontrol eder.
           [EN] Checks if data is within physical boundaries."""
        report = {}
        for sensor, limits in self.thresholds.items():
            out_of_range = df[(df[sensor] < limits['min']) | (df[sensor] > limits['max'])]
            report[sensor] = len(out_of_range)
        return report

    def check_z_scores(self, df, threshold=3):
        """[TR] Z-Skoru kullanarak istatistiksel aykırı değerleri (outliers) bulur.
           [EN] Detects statistical outliers using Z-Score."""
        outliers_report = {}
        for sensor in df.columns:
            mean = df[sensor].mean()
            std = df[sensor].std()
            z_scores = (df[sensor] - mean) / std
            outliers = df[np.abs(z_scores) > threshold]
            outliers_report[sensor] = len(outliers)
        return outliers_report