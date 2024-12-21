import textwrap

PROMPTS = {

    "default": textwrap.dedent("""
    Based on the following image captions, answer the question:
    {captions}
    Question: {question}
    Answer:
    """).strip(),

    "detailed": textwrap.dedent("""
    The image contains objects described below:
    {captions}
    Can you answer the following question based on these descriptions?
    Question: {question}
    Answer:
    """).strip(),

    "simple": textwrap.dedent("""
    {captions}
    Q: {question}
    A:
    """).strip(),
  
}

print(PROMPTS["default"])