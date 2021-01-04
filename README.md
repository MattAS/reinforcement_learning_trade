# Trading with Reinforcement learning

This repository is a small part of a bigger project designed to bring algorithmic trading to the public through methods such as Reinforcement learning, Deep Learning and Statstical forecasting.
</br>

In this project, I implemented a Deep Q-Learning Reinforcement agent to learn and perform trades in the stock market.
<br>
**Goals**:
- Create a custom enviornment filled with stock data
- Create an agent and its observation
- Train agent on the enviornment
- Test the agent's performance with a stock independent of trained stock

## Results:
The agent showed promising performance. Since the training only used one stock (Google), the model is overfitted to that stock. The training loop should be improved by either using synthetic stock market data or a culmination of multiple stocks. I should also find other metrics for the agent's observation. Some of these metrics could be the volume of trades, current time of day or day of week and fundemental data of company. The reward could also be tweaked. I should try to use relative closing prices instead of the closing price to check whether having a metric independent of the actual price helps the model learn.
