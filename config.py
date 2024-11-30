"""
config.py
"""

# Standard Imports
from typing import Literal, Optional

# Third Party Imports
from pydantic import BaseModel, Field


class Answer(BaseModel):
    question_type: Literal["multiple_choice", "open_ended", "open_ended_with_multiple_parts"] = Field(description="The type of question.")
    reasoning: str = Field(description="The reasoning for the answer.")
    final_answer: str | list[str] = Field(description="The answer to the question. For multi-part questions, returns a list of answers.")
    final_choice: Optional[int] = Field(description="The final choice of the answer (1,2,3,4) if the question is a multiple choice question.", ge=1, le=4)

SYSTEM_PROMPT = """
Answer the given math question.
You may assume there are no errors in the question.
Always reply in JSON.
If the question is a multiple choice question, you must also return the final choice.
"""