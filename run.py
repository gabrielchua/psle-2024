"""
run.py
"""

# Standard Library Imports
import csv
import os
import time
from pathlib import Path

# Third Party Imports
import instructor
import vertexai  # type: ignore
from anthropic import AnthropicBedrock
from dotenv import load_dotenv
from openai import OpenAI
from vertexai.generative_models import GenerativeModel

# Local Imports
from config import Answer, SYSTEM_PROMPT

# Load dotenv
load_dotenv()

# Initialize OpenAI client with instructor patch
openai_client = instructor.from_openai(OpenAI())

# Initialize Anthropic client with instructor patch
anthropic_client = instructor.from_anthropic(AnthropicBedrock(
    aws_region="us-west-2",
    aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_key=os.getenv("AWS_SECRET_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
))

# Initialize Google Generative AI client with instructor patch
# vertexai.init()
# gemini_client = instructor.from_vertexai(
#     client=GenerativeModel("gemini-1.5-pro-002"),
#     mode=instructor.Mode.VERTEXAI_TOOLS,
# )

# Get all image files from data folder
data_dir = Path("data")
image_files = sorted(data_dir.glob("*.png"))

models = {
    "gpt-4o-2024-11-20": openai_client,
    # "anthropic.claude-3-5-sonnet-20241022-v2:0": anthropic_client,
    # "gemini-1.5-pro-002": gemini_client,
}

for model_name, client in models.items():
    output_file = f"responses_{model_name}.csv"
    print(f"Processing {model_name}...")
    
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question_number", "model", "question_type", "reasoning", "final_answer", "choice"])
        # Process each image
        for image_path in image_files:
            retry_count = 0
            while retry_count < 2:  # Try original attempt plus one retry
                try:
                    # Extract question number from filename
                    question_number = image_path.stem
                    
                    # Create image object and get API response
                    image = instructor.Image.from_path(str(image_path))
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": image}
                        ],
                        response_model=Answer,
                        max_tokens=4000,
                        temperature=0.1,
                    )
                    
                    # Write response data to CSV
                    writer.writerow([
                        question_number,
                        model_name,
                        response.question_type,
                        response.reasoning,
                        response.final_answer,
                        response.final_choice or ""  # Handle optional choice
                    ])
                    
                    # Print response details after each question
                    print(f"\nQuestion {question_number} ({model_name}):")
                    print(f"Reasoning: {response.reasoning}")
                    print(f"Answer: {response.final_answer}")
                    if response.final_choice:
                        print(f"Choice: {response.final_choice}")
                    break
                    
                except Exception as e:
                    print(f"Error processing question {image_path.stem}: {str(e)}")
                    if retry_count < 1:  # Only wait if we're going to retry
                        print("Waiting 1 minute before retrying...")
                        time.sleep(60)
                    retry_count += 1
            
            if retry_count == 2:
                print(f"Skipping question {image_path.stem} after failed retry")
                writer.writerow([
                    image_path.stem,
                    model_name,
                    "",
                    "ERROR - Skipped after retry",
                    "",
                    ""
                ])
