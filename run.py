"""
run.py
"""

# Standard Library Imports
import base64
import csv
import json
import os
import time
from pathlib import Path

# Third Party Imports
import instructor
import vertexai
from anthropic import AnthropicBedrock
from dotenv import load_dotenv
from openai import OpenAI
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
)

# Local Imports
from config import Answer, GEMINI_RESPONSE_SCHEMA, GEMINI_SAFETY, SYSTEM_PROMPT


# Constants
MAX_RETRIES = 3
TEMPERATURE = 0.1
MAX_TOKENS = 8192
RETRY_WAIT = 60

# Load dotenv
load_dotenv()

# Initialize OpenAI client with instructor patch
openai_client = instructor.from_openai(OpenAI())

# Initialize Anthropic client with instructor patch
anthropic_client = instructor.from_anthropic(
    AnthropicBedrock(
        aws_region="us-west-2",
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    )
)

# Initialize Google Generative AI client
vertexai.init()

gemini_model = GenerativeModel(
    "gemini-1.5-pro-002",
    system_instruction=[SYSTEM_PROMPT],
    safety_settings=GEMINI_SAFETY,
)


def base64_encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Get all image files from data folder
data_dir = Path("data")
image_files = sorted(data_dir.glob("*.png"))

models = {
    "gpt-4o-2024-11-20": openai_client,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": anthropic_client,
    "gemini-1.5-pro-002": gemini_model,
}

for model_name, client in models.items():
    output_file = f"responses_{model_name}.csv"
    markdown_file = f"responses_{model_name}.md"
    print(f"Processing {model_name}...")

    with open(output_file, "w", newline="") as csvfile, open(
        markdown_file, "w"
    ) as mdfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "question_number",
                "model",
                "question_type",
                "reasoning",
                "final_answer",
                "choice",
            ]
        )

        mdfile.write(f"# Responses for {model_name}\n\n")

        # Process each image
        for image_path in image_files:
            # Extract question number from filename
            question_number = image_path.stem

            # For Gemini
            if "gemini" in model_name:

                for attempt in range(MAX_RETRIES):
                    try:
                        image = Part.from_data(
                            mime_type="image/png",
                            data=base64.b64decode(base64_encode_image(image_path)),
                        )
                        response = client.generate_content(
                            ["Here is the question", image],
                            generation_config=GenerationConfig(
                                response_mime_type="application/json",
                                response_schema=GEMINI_RESPONSE_SCHEMA,
                            ),
                        )
                        response = json.loads(
                            response.candidates[0].content.parts[0].text
                        )[0]
                        question_type = response["step_1_question_type"] or None
                        reasoning = response["step_2_reasoning"] or None
                        final_answer = response["step_3_final_answer"] or None
                        final_choice = response["step_4_final_choice"] or None
                        break
                    except Exception as e:
                        try:
                            print(response)
                        except:
                            pass
                        print(f"Error: {e}")
                        if attempt < 2:  # Don't sleep on last attempt
                            time.sleep(RETRY_WAIT)
                        else:
                            # Skip after 2 failed attempts
                            print(f"Skipping {question_number} after 2 failed attempts")
                            question_type = "skipped"
                            reasoning = f"Skipped due to errors - {e}"
                            final_answer = "skipped"
                            final_choice = None
                            break

            # For OpenAI and Anthropic
            else:
                for attempt in range(MAX_RETRIES):
                    try:
                        image = instructor.Image.from_path(str(image_path))

                        # Create image object and get API response
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": image},
                            ],
                            response_model=Answer,
                            max_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE,
                        )

                        question_type = response.question_type
                        reasoning = response.reasoning
                        final_answer = response.final_answer
                        final_choice = response.final_choice
                        break
                    except Exception as e:
                        if attempt < 2:  # Don't sleep on last attempt
                            time.sleep(RETRY_WAIT)
                        else:
                            raise e  # Re-raise exception on final attempt

            # Write response data to CSV
            writer.writerow(
                [
                    question_number,
                    model_name,
                    question_type,
                    reasoning,
                    final_answer,
                    final_choice or "",  # Handle optional choice
                ]
            )

            # Write to markdown file
            mdfile.write(f"\n## Question {question_number}\n\n")
            mdfile.write(f"**Question Type:** {question_type}\n\n")
            mdfile.write(f"**Reasoning:**\n{reasoning}\n\n")
            mdfile.write(f"**Answer:** {final_answer}\n")
            if final_choice:
                mdfile.write(f"\n**Choice:** {final_choice}\n")
            mdfile.write("\n---\n")

            # Print response details after each question
            print("\n##########################")
            print(f"Question {question_number} ({model_name}):")
            print(f"Reasoning: {reasoning}")
            print(f"Answer: {final_answer}")
            if final_choice:
                print(f"Choice: {final_choice}")
