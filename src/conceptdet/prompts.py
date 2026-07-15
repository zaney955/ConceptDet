from __future__ import annotations

SYSTEM_PROMPT = "You are a precise visual localization assistant."


# Keep this user-side protocol compatible with the prompt used to fine-tune
# ConceptSeg-R1. Whitespace and wording are intentionally preserved because
# this checkpoint can switch candidates after seemingly harmless prompt edits.
REFERENCE_GUIDED_QUESTION_TEMPLATE = (
    "\n"
    'Your task is to locate the object matching "{question}" in the Target Image.\n'
    "Data provided:\n"
    "1. Reference Image {ref_bboxes}.\n"
    "2. Target Image: The image to locate.\n"
    "Think through the reasoning process in your mind， induce the visual rule "
    "{check_prompt}, apply this rule to locate the corresponding object in the Target Image.\n"
    "Finally,  provide the bounding box  and a 1-2 word  noun phrase for the object in the "
    "target image. \n"
    "Output strictly in the following format: <think>[Your step-by-step analysis and "
    "reasoning]</think>  {check_answer} <bbox>[x3, y3, x4, y4]</bbox> "
    "<answer>concise noun phrase for target object</answer>\n"
    "     "
)


def build_problem(query: str, reference_box_count: int, input_size: int = 600) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if reference_box_count < 1:
        raise ValueError("reference_box_count must be positive")
    if input_size < 1:
        raise ValueError("input_size must be positive")
    return REFERENCE_GUIDED_QUESTION_TEMPLATE.format(
        question=query,
        ref_bboxes=": Bounding boxes at red-marked boxs",
        check_prompt="",
        check_answer="<rule>Visual rule of the reference targets in the reference image</rule>",
    )


def build_messages(problem: str) -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "text": None},
                {"type": "image", "text": None},
                {"type": "text", "text": problem},
            ],
        },
    ]
