# PSLE 2024

Here we evaluate the performance of different LLMs on the PSLE 2024 questions.

## Results
| Model                                      | Total Score |
|--------------------------------------------|-------------|
| gpt-4o-2024-11-20                          | 48          |
| anthropic.claude-3-5-sonnet-20241022-v2:0  | 52          |
| gemini-1.5-pro-002                         | TBD         |

## Prompt
See `config.py` for the system prompt used.

## Setup
1. Download the data from [Hugging Face](https://huggingface.co/datasets/gabrielchua/psle-2024).

2. Install the dependencies.

```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add the following environment variables:

```
AWS_ACCESS_KEY_ID=<your-aws-access-key-id>
AWS_SECRET_KEY=<your-aws-secret-key>
AWS_SESSION_TOKEN=<your-aws-session-token>
VERTEX_AI_PROJECT=<your-vertex-ai-project>
```

4. Run the script.

```bash
python run.py
```

## Note
You may find a PDF copy of the questions [here](https://www.mendaki.org.sg/resources/?mode=exams-paper).

This repository and its contents are intended solely for research and educational purposes. The PSLE questions used in this analysis remain the intellectual property of their respective copyright holders (e.g. Singapore Examinations and Assessment Board or the authors of the questions). This work is not affiliated with, endorsed by, or sponsored by any educational institution or examination board. The use of these questions is for academic research to evaluate AI language models' capabilities and does not claim any rights over the original material.

Users are reminded to respect intellectual property rights and to use this resource responsibly. Any commercial use or redistribution of the original examination questions is strictly prohibited.
