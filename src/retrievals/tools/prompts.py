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
