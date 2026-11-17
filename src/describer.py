"""
[TR] Kural tabanlı kararı LLM ile açıklar. Kararı LLM vermez, değiştiremez.
[EN] Uses an LLM to explain the rule-based verdict. The LLM does not make or change the verdict.
"""
import json
import os

from dotenv import load_dotenv


def build_prompt(summary):
    return f"""You are a Data Quality Engineer at a smart factory.
Automated checks were run on a temperature sensor stream:
- range: physical limit check (0-150 °C)
- robust_z: modified Z-score with median/MAD, |z| > 3.5
- frozen: value unchanged for >= 10 consecutive readings (stuck sensor)

Results (JSON):
{json.dumps(summary, indent=2)}

The verdict "{summary['verdict']}" was decided by a fixed rule ({summary['verdict_rule']}).
It is final. Do NOT change, question or re-decide it.
Explain in a few short paragraphs what the flagged rows likely mean physically,
how they would affect an ML model trained or run on this data, and what to do next."""


class DataQualityDescriber:
    def __init__(self, model="gpt-4o-mini"):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (.env)")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def explain(self, summary):
        """[TR] Kararı açıklar.  [EN] Explains the verdict."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": build_prompt(summary)}],
        )
        return response.choices[0].message.content
