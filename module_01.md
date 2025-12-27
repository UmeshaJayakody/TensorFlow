# Different Types of Learnings

## Supervised Learning (Learning with answers)

## Idea
The computer is taught, like a student in class.

## How it works
- We give data + correct answer
- Computer learns the connection
- Later, it can answer new questions

## Think like this
Teacher shows a question and the correct answer.

## Example
- Photo + label → cat
- Photo + label → dog
- After many examples, computer can say: this is a cat


## Unsupervised Learning (Learning without answers)

## Idea
The computer explores by itself.

## How it works
- We give only data
- No answers, no labels
- Computer finds similar things

## Think like this
Give many toys and ask: "put similar toys together"

## Example
- Customer data → computer groups similar customers
- Songs → computer groups similar songs



## Reinforcement Learning (Learning by reward)

## Idea
The computer learns by doing.

## How it works
- Computer takes an action
- Good result → reward
- Bad result → punishment
- It remembers and improves

## Think like this
Training a dog:
- Sit → treat
- Wrong → no treat

## Example
- Game AI learns how to win
- Robot learns how to walk



## Very short memory trick
- Supervised → answers given
- Unsupervised → answers not given
- Reinforcement → reward given

## How models like Gemini are trained

### Self-supervised learning / Unsupervised learning
First, the model learns from large amounts of text/images without labels by predicting missing or next words.

This builds general knowledge about language and patterns.

This is basically a form of unsupervised learning (the model teaches itself).


### Supervised learning (fine-tuning)
After the base training, the model is often fine-tuned with examples + correct answers (like question and good response pairs).

This helps it follow instructions better.


### Reinforcement learning from human feedback (RLHF)
Then humans rate answers, and the model learns which answers are better using reward signals.

This is a type of reinforcement learning.


So overall, these models use a mix of learning techniques — mainly self-supervised learning plus supervised fine-tuning and reinforcement learning from human feedback — not just one simple category.