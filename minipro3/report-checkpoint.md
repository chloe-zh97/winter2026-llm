## Baseline vs. Single Agent Performance
For Baseline testing, I tested the following cases:

- "What is the P/E ratio of Apple (AAPL)?"
  - **Easy** question
  - Baseline cannot provide real-time data of the question.
  - Single agent for this easy question, it can directly give a straightfoward answer.
  - Single agent's performance is better than Baseline.

- "Which energy stocks in the database had the best 6-month performance?"
  - Medium question, requires sector lookup + price fetch
  - Baseline cannot access the data and is not able to provide the answer.
  - Signle agent uses two tools: **get_tickers_by_sector** and **get_price_performance**, and returns a list of the stocks performance.

- "Top 3 tech stocks that dropped this month but grew this year."
  - Hard multi-condition question
  - Baseline failed to answer the question
  - Single agent uses three tools: **get_tickers_by_sector**, **get_price_performance**, **get_price_performance**. And has the capability to organize the answer.

## Why we need multi-agent system for this application?
We need a multi-agent system because the task is complex and involves different types of information: stock prices, fundamentals, and news sentiment. A single agent might make mistakes or hallucinate when using all tools at once. By splitting into multiple specialized agents, each agent focuses on one type of data, works more accurately, and can solve multi-step questions step by step. The agents share context, so they can still answer complex questions together.

## The design being considered for the multiagent and why?
The multi-agent design uses several specialized agents, each with access to only the tools they need. This makes each agent more focused and accurate. For example, one agent handles stock fundamentals, another handles news sentiment, and another handles price performance. The main reason for this design is to reduce errors and hallucinations, because smaller, focused agents are easier for the LLM to control than one large agent with all tools. The agents communicate through a shared context, so complex questions can be split and solved step by step.
