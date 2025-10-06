import logging
import re
logger = logging.getLogger(__name__)


def str_difference(str1, str2):
    """Quick difference analysis between two strings."""
    logger.info(f"Lengths: {len(str1)} vs {len(str2)}")
    if len(str1) != len(str2):
        logger.info("Strings have different lengths!")

    # Show unique characters
    unique_chars = set(str1) ^ set(str2)  # symmetric difference
    if unique_chars:
        logger.info(f"Unique characters: {unique_chars}")

    # Show first difference
    for i, (c1, c2) in enumerate(zip(str1, str2)):
        if c1 != c2:
            logger.info(f"First difference at position {i}:")
            logger.info(f"str1: '{repr(c1)}' in context: ...{str1[i - 10:i + 10]}...")
            logger.info(f"str2: '{repr(c2)}' in context: ...{str2[i - 10:i + 10]}...")
            break


async def str_difference_llm(original_query, corrected_query, assistant):
    """Quick difference analysis between two strings."""
    prompt = f"""You are given in input two SQL queries, one is the original query, the other one its correction.
    Can you briefly explain what are the changes in the corrected query?
    Original query: {original_query} \n
    Corrected query: {corrected_query} \n
    """

    assistant.add_message(role="user", message=prompt)
    answer = await assistant.get_response()

    logger.info(answer)


def lowercase_placeholder_values(sql_text: str) -> str:
    """
    Lowercase values in placeholders of the form [entity_type@value].

    Args:
        sql_text: SQL query text containing placeholders

    Returns:
        Modified SQL text with lowercased values in placeholders
    """
    pattern = r"\[([a-zA-Z_]+)@([a-zA-Z0-9_/\-\(\)\'\\ ]+)\]"

    def lowercase_match(match):
        entity_type = match.group(1)
        value = match.group(2).lower()
        return f"[{entity_type}@{value}]"

    return re.sub(pattern, lowercase_match, sql_text)
