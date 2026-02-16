"""
[TR] İstatistiksel veri hatalarını analiz eder ve anlamlı raporlar sunar.
[EN] Analyzes statistical data errors and provides meaningful reports.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

class DataQualityDescriber:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def describe_errors(self, range_errors, z_errors):
        """[TR] Hata raporlarını yorumlar. [EN] Interprets error reports."""
        prompt = f"""
        You are a Data Quality Engineer at a smart factory. 
        A sensor validation check has been performed:
        - Out-of-range errors: {range_errors}
        - Z-Score outliers (3-sigma): {z_errors}

        Explain the potential impact of these data quality issues on a machine learning model.
        Suggest if the data is 'SAFE' or 'UNSAFE' for training/inference.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content