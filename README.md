# Generating-Traffic-Flows-with-MARL
# Accepted by AROB 2021
I will upload my paper later.
# Feature extraction and image inverse generation
![image](https://github.com/ZHONGJunjie86/Mixed_Input_PPO_CNN_LSTM_Car_Navigation/blob/master/result/old/img_generante.JPG)
![image](https://github.com/ZHONGJunjie86/Mixed_Input_PPO_CNN_LSTM_Car_Navigation/blob/master/result/image%20inverse%20generation.gif)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-Mixed_Input_AC_CNN_LSTM_Car_Navigation/blob/main/training_data/2.jpg)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-Mixed_Input_AC_CNN_LSTM_Car_Navigation/blob/main/training_data/3.jpg)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-Mixed_Input_AC_CNN_LSTM_Car_Navigation/blob/main/training_data/4.jpg)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-Mixed_Input_AC_CNN_LSTM_Car_Navigation/blob/main/training_data/5.jpg)
![image](https://github.com/ZHONGJunjie86/Generating-Traffic-Flows-with-Mixed_Input_AC_CNN_LSTM_Car_Navigation/blob/main/training_data/6.jpg)
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
