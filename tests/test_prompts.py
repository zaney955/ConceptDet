from conceptdet.prompts import SYSTEM_PROMPT, build_messages, build_problem


def test_reference_guided_problem_matches_conceptseg_user_protocol() -> None:
    problem = build_problem(" matching bolt ", 2, input_size=600)
    expected = (
        "\n"
        'Your task is to locate the object matching "matching bolt" in the Target Image.\n'
        "Data provided:\n"
        "1. Reference Image : Bounding boxes at red-marked boxs.\n"
        "2. Target Image: The image to locate.\n"
        "Think through the reasoning process in your mind， induce the visual rule , apply "
        "this rule to locate the corresponding object in the Target Image.\n"
        "Finally,  provide the bounding box  and a 1-2 word  noun phrase for the object in "
        "the target image. \n"
        "Output strictly in the following format: <think>[Your step-by-step analysis and "
        "reasoning]</think>  <rule>Visual rule of the reference targets in the reference "
        "image</rule> <bbox>[x3, y3, x4, y4]</bbox> <answer>concise noun phrase for target "
        "object</answer>\n"
        "     "
    )
    assert problem == expected


def test_system_prompt_is_preserved() -> None:
    problem = build_problem("bolt", 1)
    messages = build_messages(problem)
    assert SYSTEM_PROMPT == "You are a precise visual localization assistant."
    assert messages[0]["content"][0]["text"] == SYSTEM_PROMPT
    assert messages[1]["content"][:2] == [
        {"type": "image", "text": None},
        {"type": "image", "text": None},
    ]
