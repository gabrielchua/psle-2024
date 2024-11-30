"""
config.py
"""

# Standard Imports
from typing import Literal, Optional

# Third Party Imports
from pydantic import BaseModel, Field


class Answer(BaseModel):
    question_type: Literal[
        "multiple_choice", "open_ended", "open_ended_with_multiple_parts"
    ] = Field(description="The type of question.")
    reasoning: str = Field(description="The reasoning for the answer.")
    final_answer: str | list[str] = Field(
        description="The answer to the question. For multi-part questions, returns a list of answers."
    )
    final_choice: Optional[int] = Field(
        description="The final choice of the answer (1,2,3,4) if the question is a multiple choice question.",
        ge=1,
        le=4,
    )


SYSTEM_PROMPT = """
Answer the given math question.
You may assume there are no errors in the question.
Always reply in JSON.
If the question is a multiple choice question, you must also return the final choice.
"""

GEMINI_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "step_1_question_type": {
                "type": "string",
                "enum": [
                    "multiple_choice",
                    "open_ended",
                    "open_ended_with_multiple_parts",
                ],
                "description": "The type of question.",
                "nullable": False,
            },
            "step_2_reasoning": {
                "type": "string",
                "description": "The reasoning for the answer.",
                "nullable": False,
            },
            "step_3_final_answer": {
                "type": "string",
                "description": "The final answer.",
                "nullable": False,
            },
            "step_4_final_choice": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4,
                "description": "The final choice of the answer (1,2,3,4) if the question is a multiple choice question.",
                "nullable": True,
            },
        },
    },
}

GEMINI_SAFETY = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF,
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF,
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF,
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF,
    ),
]
