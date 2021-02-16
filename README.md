# Generating-Traffic-Flows-with-MARL
# Accepted by AROB 2021
Find my paper [Generation of Traffic Flows in Multi-Agent Traffic Simulation with Agent Behavior Model based on Deep Reinforcement Learning](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-MARL/blob/main/OS10-6.pdf).
# Abstract
In multi-agent based traffic simulation, agents are always supposed to move following existing instructions, and mechanically and unnaturally imitate human behaviour. The human drivers perform acceleration or deceleration irregularly all the time, which seems unnecessary in some conditions. For letting agents in traffic simulation behave more like humans and recognize other agents’ behaviour in complex conditions, we propose a unified mechanism for agents learn to decide various accelerations by using deep reinforcement learning based on a combination of regenerated visual images revealing some notable features, and numerical vectors containing some important data such as instantaneous speed. By handling batches of sequential data, agents are enabled to recognize surrounding agents’ behaviour and decide their own acceleration. In addition, we can generate a traffic flow behaving diversely to simulate the real traffic flow by using an architecture of fully decentralized training and fully centralized execution without violating Markov assumptions.
# Feature extraction and image inverse generation
![image](https://github.com/ZHONGJunjie86/Mixed_Input_PPO_CNN_LSTM_Car_Navigation/blob/master/result/old/img_generante.JPG)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-MARL/blob/main/training_data/Conference_1.gif)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-MARL/blob/main/training_data/Conference_2.gif)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-Mixed_Input_AC_CNN_LSTM_Car_Navigation/blob/main/training_data/2.jpg)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-Mixed_Input_AC_CNN_LSTM_Car_Navigation/blob/main/training_data/3.jpg)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-Mixed_Input_AC_CNN_LSTM_Car_Navigation/blob/main/training_data/4.jpg)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-Mixed_Input_AC_CNN_LSTM_Car_Navigation/blob/main/training_data/5.jpg)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-Mixed_Input_AC_CNN_LSTM_Car_Navigation/blob/main/training_data/7.jpg)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-Mixed_Input_AC_CNN_LSTM_Car_Navigation/blob/main/training_data/8a.jpg)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-Mixed_Input_AC_CNN_LSTM_Car_Navigation/blob/main/training_data/8b.jpg)
### TD
<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangledown&space;Advantage&space;=&space;\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}(r_{t}&plus;V_{s&plus;1}^{n}-V_{s}^{n})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangledown&space;Advantage&space;=&space;\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}(r_{t}&plus;V_{s&plus;1}^{n}-V_{s}^{n})" title="\bigtriangledown&space;Advantage&space;=&space;\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}(r_{t}&plus;V_{s&plus;1}^{n}-V_{s}^{n})" /></a>
### MC
<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangledown&space;Advantage&space;=&space;\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}(R_{t}-V_{s}^{n})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangledown&space;Advantage&space;=&space;\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}(R_{t}-V_{s}^{n})" title="\bigtriangledown&space;Advantage&space;=&space;\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}(R_{t}-V_{s}^{n})" /></a>
### Actor Critic (TD)
　<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangledown&space;R&space;=&space;\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}(r_{t}&plus;V_{s&plus;1}^{n}-V_{s}^{n})\bigtriangledown&space;log&space;P_{\Theta&space;}(a_{t}^{n}|s_{t}^{n})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangledown&space;R&space;=&space;\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}(r_{t}&plus;V_{s&plus;1}^{n}-V_{s}^{n})\bigtriangledown&space;log&space;P_{\Theta&space;}(a_{t}^{n}|s_{t}^{n})" title="\bigtriangledown R = \frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}(r_{t}+V_{s+1}^{n}-V_{s}^{n})\bigtriangledown log P_{\Theta }(a_{t}^{n}|s_{t}^{n})" /></a>
# About GAMA
　The GAMA is a platefrom to do simulations.      
