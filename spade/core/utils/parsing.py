"""Response parsing and generated-code repair helpers."""

import ast
import re


def extract_boxed_answer(solution: str) -> str | None:
    if not solution:
        return None
    # Models may mention a candidate answer during reasoning before emitting
    # their final answer. Parse balanced braces from the *last* \boxed marker so
    # nested LaTeX (for example \frac{1}{2}) and final-answer semantics both work.
    boxed_positions = [match.start() for match in re.finditer(r"\\boxed", solution)]
    for boxed_index in reversed(boxed_positions):
        start = solution.find("{", boxed_index + len("\\boxed"))
        if start < 0:
            continue
        depth = 0
        for index in range(start, len(solution)):
            if solution[index] == "{":
                depth += 1
            elif solution[index] == "}":
                depth -= 1
                if depth == 0:
                    content = solution[start + 1 : index].strip()
                    for _ in range(2):
                        if content.startswith("{") and content.endswith("}"):
                            content = content[1:-1].strip()
                    return content
    return None


def validate_boxed_format(answer: str) -> bool:
    return extract_boxed_answer(answer) is not None


class LanguageGameReward:
    success_reward = 1.0
    format_error_reward = -1.0
    invalid_action_reward = -0.5
    partial_success_reward = 0.5
    hint_penalty = -0.1
    step_penalty = -0.01


def format_error_response(task_suffix: str) -> str:
    return (
        "Invalid format. Please use \\boxed{your_answer}. "
        "For example, if the answer is 42, respond with \\boxed{42}.\n\n" + task_suffix
    )


def extract_tool_call(raw_action: str) -> str | None:
    answer_match = re.search(r"<answer>(.*?)</answer>", raw_action, re.DOTALL)
    if answer_match:
        return f"ANSWER: {answer_match.group(1).strip()}"
    tool_match = re.search(r"<tool_call>(.*?)</tool_call>", raw_action, re.DOTALL)
    return tool_match.group(1).strip() if tool_match else None


def extract_command(raw_action: str) -> str | None:
    answer_match = re.search(r"<answer>(.*?)</answer>", raw_action, re.DOTALL)
    if answer_match:
        return f"ANSWER: {answer_match.group(1).strip()}"
    command_match = re.search(r"<command>(.*?)</command>", raw_action, re.DOTALL)
    return command_match.group(1).strip() if command_match else None


def parse_action(raw_action: str, action_format: str = "boxed") -> str:
    """Extract the final visible action while preserving environment wrappers.

    Multi-turn models commonly emit hidden ``<think>`` text and may mention an
    earlier candidate action. The environment should receive the final visible
    boxed/tagged action, not the complete reasoning transcript.
    """
    if not isinstance(raw_action, str):
        raw_action = str(raw_action)
    visible = raw_action.rsplit("</think>", 1)[-1].strip()

    if action_format == "boxed":
        extracted = extract_boxed_answer(visible)
        return f"\\boxed{{{extracted}}}" if extracted is not None else visible

    if action_format == "tool_call":
        matches = list(
            re.finditer(
                r"<(tool_call|answer)>(.*?)</\1>",
                visible,
                re.DOTALL | re.IGNORECASE,
            )
        )
        if not matches:
            return visible
        match = matches[-1]
        tag = match.group(1).lower()
        return f"<{tag}>{match.group(2).strip()}</{tag}>"

    if action_format == "command":
        matches = list(
            re.finditer(
                r"<(command|answer)>(.*?)</\1>",
                visible,
                re.DOTALL | re.IGNORECASE,
            )
        )
        if not matches:
            return visible
        match = matches[-1]
        tag = match.group(1).lower()
        return f"<{tag}>{match.group(2).strip()}</{tag}>"

    raise ValueError(
        f"Unsupported action_format={action_format!r}; expected 'boxed', 'tool_call', or 'command'"
    )


def repair_fstring_braces(code: str, max_iter: int = 20) -> str:
    for _ in range(max_iter):
        try:
            ast.parse(code)
            return code
        except SyntaxError as exc:
            if not exc.lineno:
                return code
            lines = code.split("\n")
            line_index = exc.lineno - 1
            if line_index >= len(lines):
                return code

            def fix(match: re.Match) -> str:
                inner = match.group(1)
                try:
                    ast.parse(inner, mode="eval")
                    return match.group(0)
                except SyntaxError:
                    return "{{" + inner + "}}"

            repaired = re.sub(r"\{([^{}]*)\}", fix, lines[line_index])
            if repaired == lines[line_index]:
                return code
            lines[line_index] = repaired
            code = "\n".join(lines)
    return code


def extract_game_code(response_text: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    return repair_fstring_braces(match.group(1) if match else response_text)
