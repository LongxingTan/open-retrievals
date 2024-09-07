"""Some default prompts"""

RAG_PROMPT = """
"""


SUMMARIZE_PROMPT = """
Generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description.

#####
Entities: {entity_name}
Description List: {description_list}
#####
Output:
"""


QUERY_GENERATION_PROMPT = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""
