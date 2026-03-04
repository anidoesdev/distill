"""Extraction prompt templates.

IMPORTANT: This prompt is part of the training artifact. It is used verbatim
during teacher dataset generation (sessions 4-5). Any change to the wording
after dataset generation invalidates the training distribution.

Do NOT generate this string dynamically from the schema at runtime. The exact
text must be frozen before generating training data.
"""


# The system prompt passed as {"role": "system", "content": ...}
EXTRACTION_SYSTEM_PROMPT = """\
You are a scientific information extractor. Given a section of a scientific paper, \
extract structured information and return ONLY a valid JSON object. \
No explanation, no markdown, no surrounding text.

The JSON must have exactly these six keys:
{
  "authors": ["list of author full names as stated in the paper"],
  "methodology": "one to three sentence description of methods or algorithms used",
  "datasets_used": ["list of dataset or benchmark names"],
  "key_findings": ["list of main results, one sentence each"],
  "limitations": ["list of limitations or constraints acknowledged by the authors"],
  "statistical_tests": ["list of statistical tests or significance measures mentioned"]
}

Rules:
- Return empty list [] for any field not mentioned in the text.
- Return empty string "" for methodology if no methods are described.
- Do not infer or assume information not explicitly stated in the text.
- Author names must appear verbatim as written in the text.
- Output ONLY the JSON object. Nothing before or after it.\
"""


def build_messages(section_text: str) -> list[dict[str, str]]:
    """Build the message list for apply_chat_template or the OpenAI API format.

    This is the single canonical way to construct the extraction prompt.
    Both training data generation and production inference must call this function.
    """
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Extract structured information from this paper section:\n\n{section_text.strip()}",
        },
    ]
