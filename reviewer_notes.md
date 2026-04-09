# Reviewer Notes

## Why I chose Mistral

I chose **Mistral** as the primary LLM for this project mainly because of **cost and practicality constraints** during development.

For this assignment, I needed a model that could support:

- tool use / function-calling style workflows
- repeated multi-step research runs
- experimentation with memory compaction and retrieval budgets
- enough iterations to refine the prototype without making every run expensive

I considered alternatives such as **OpenAI**, **Claude**, and **Gemini**, but for this project they were less attractive from a cost-to-iteration perspective. Since this task involved many repeated runs while tuning decomposition, retrieval, summarization, and offline/demo behavior, I chose the option that let me iterate more comfortably within budget.

## Why this still fits the assignment

The assignment asked for:

- an LLM with tool use capability
- a memory strategy under explicit constraints
- workflow orchestration such as n8n / Dify

Using Mistral still satisfies that requirement set. The key design focus of this submission is not the brand of model, but the **research-agent architecture** around it:

- decomposition into sub-questions
- evidence gathering through tools
- bounded episodic memory
- summarization cascade under memory pressure
- budgeted retrieval for final answer synthesis

## Practical takeaway

The model choice was a **budget-aware engineering decision**. I wanted to spend effort demonstrating:

- constraint handling
- reproducibility
- architecture trade-offs
- orchestration and memory design

rather than spending most of the project budget on model calls alone.
