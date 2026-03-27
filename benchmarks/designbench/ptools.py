"""Interfaces for DesignBench code generation."""

import re

from secretagent.core import interface


@interface
def generate_code(reference_html: str, framework: str, metadata: dict) -> str:
    """You are an expert HTML/CSS developer.
    You take screenshots of a reference web page from the user, and then build single page apps.

    - Make sure the app looks exactly like the screenshot.
    - Pay close attention to background color, text color, font size, font family, padding, margin, border, etc. Match the colors and sizes exactly.
    - Use the exact text from the screenshot.
    - Do not add placeholder comments in place of real code. Write the full code.
    - Repeat elements as needed to match the screenshot.
    - For images, use placeholder images from https://placehold.co and include detailed alt text.

    Please return code inside a markdown code block appropriate for `framework`.
    Do not output any extra information or comments.
    """


def _extract_code_block(text: str, framework: str) -> str:
    fence_map = {
        'vanilla': 'html',
        'react': 'jsx',
        'vue': 'vue',
        'angular': 'angular',
    }
    lang = fence_map.get(framework, 'html')

    match = re.search(rf"```{lang}\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()


def extract_code(response: str, framework: str) -> str:
    """Extract code block from model output."""
    return _extract_code_block(response, framework)
