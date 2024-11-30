"""
schema.py
"""

# Standard Imports
from typing import Optional

# Third Party Imports
from pydantic import BaseModel, Field


class Answer(BaseModel):
    reasoning: str = Field(description="The reasoning for the answer.")
    answer: str = Field(description="The answer to the question.")
    final_choice: Optional[str] = Field(description="The final choice of the answer, if the question is a multiple choice question.")
