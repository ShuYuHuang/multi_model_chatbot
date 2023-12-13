######
# Prompt Template list for different functions
######
from langchain.prompts import PromptTemplate

__all__ = [
    'QUERY_INTEGRATION_TEMPLATE',
    'DATA_SHORT_DESCRIPTION',
    'INITIAL_CSV_PLOT',
    'PRMPTED_CSV_PLOT'
]
# ------------------------------------------------------------------------------------------
QUERY_INTEGRATION_TEMPLATE  = """
Request:

The questions are all related to either query results or history, 
please reply to the asked questions only, do not extend to other questions

If the question is related to querying results:
- Extract information from queried result and answer the question
Else if the question is related to history:
- Extract information from history and answer the question

Queried results form database:
```
{queried}
```

History:
```
{history}
```

You are asked:
{input}

Respond:
"""

# ------------------------------------------------------------------------------------------
DATA_SHORT_DESCRIPTION  = """
Request:
The following is a table data, which is used to help with explaination some trends.

Table data:
```
{table_data}
```

Give a Short Description fo the Data.
Respond:
"""


# INITIAL_CSV_PLOT  = """
# Request:
# This is the head 3 lines of my CSV file with the filename {filename},
# it was loaded as pandas DataFram csv_df:
# ```
# {head3lines}
# ```

# According to these information, 
# generate the code <code> for plotting the previous data in plotly, in the format requested. 
# The solution should be given using plotly and only plotly. Do not use matplotlib.
# Return the code <code> in the following format ```python <code>```
# """
# ------------------------------------------------------------------------------------------
# --Version 1 of csv plot--
PRMPTED_CSV_PLOT = """
You are a perfect data scientist masters python.

This is the part of my CSV file:
```
{head3lines}
```
It have been saved as DataFrame csv_df.

Aim of investigation:
```{project_aim}```

Instruction:
```{instructions}```

Generate the harmless simple code CODE deal with csv_df according to the requirements.
Here are some cases that may used to correct the output codes:
CASE 1: If the instructions is for plotting:
    Dont use pd.read_csv, use variable csv_df directly
    Don't use pivot_table.
    Mind the spaces included in the field name of plot.
    The solution should be given using plotly and only plotly.
    Do not use matplotlib.
    result should be stored in variable 'fig'
CASE 2: If the instructions is for text or table generation:
    Instead of using pd.read_csv, use variable csv_df directly
    The result of code should be stored in a markdown string format called 'resp'
    Don't do anything related to 'resp' if it is for plotting

Return the code CODE in the following format:
```python (started from "```python")
CODE
``` (ended by "```")
"""


## --Version 2 of csv plot--
# PRMPTED_CSV_PLOT  = """
# Let's decode the way to respond to the queries.
# The responses depend on the type of information requested in the query. 

# 1. If the query requires a table, format your answer like this:
#     {{"table": {{"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}}}

# 2. For a bar chart, respond like this:
#     {{"bar": {{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}}}

# 3. If a line chart is more appropriate, your reply should look like this:
#     {{"line": {{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}}}

# Note: We only accommodate two types of charts: "bar" and "line".

# 4. For a plain question that doesn't need a chart or table, your response should be:
#     {{"answer": "Your answer goes here"}}

# For example:
#     {{"answer": "The Product with the highest Orders is '15143Exfo'"}}

# 5. If the answer is not known or available, respond with:
#     {{"answer": "I do not know."}}

# Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
# For example: {{"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}}

# Now, let's tackle the query step by step. Here's the query for you to work on:
# {instructions}
# """



INITIAL_PLOT_INSTRUCTION = """
Plot correlation map, and plot line chart
"""

DISCRIBE_PLOT = """
According to request {user_input}, give discription of this <image>
"""

combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
Source: 28-pl
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
Source: 30-pl
Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
Source: 4-pl
=========
FINAL ANSWER: This Agreement is governed by English law.
SOURCES: 28-pl

QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
Source: 0-pl
Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
Source: 24-pl
Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
Source: 5-pl
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
Source: 34-pl
=========
FINAL ANSWER: The president did not mention Michael Jackson.
SOURCES:

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)

prompt_template = """Write a concise summary of the following:


"{text}"


CONCISE SUMMARY:"""
SUMMARY_PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])