# rPGLU: recurrent Potential-Gated Linear Unit (WIP)
The main achivement of this work is the rPGLU. Currently named SLGU in repo and exists in the `/SGLU` folder. 

Hypothesis: The behaviour of potential in biological neurons capture interesting timing dynamics, can we replicate this behaviour and increase the performance of Artifical Neural Networks without full biological modelling as in Spiking Neural Networks?

I used the Tiselac timeseries dataset for development: https://www.timeseriesclassification.com/description.php?Dataset=Tiselac

Mulitple approaches have been tried, but this is a brief description of the latest version of rPGLU:
- The hidden state of the recurrent network is the potential vector.
- Potential for each neuron builds up or dissapears over time based on input values.
- If the potential goes above a treshold, the potential is sent trough to the next layer, otherwise it is 0.
- This gating or filtering based on the treshold results in non-differentiable activation functions, thus a mathemtical bump function is used to approximate the gradient in custom surrogate functions for the backwards passes. 

Result:
- (1) With a final MLP on all 23 timesteps on both the rPGLU network and the LSTM network, the rPGLU matches the accuracy of LSTMs with 1%.
- (2) With a final MLP on the last timestep only, the rPGLU network falls 5% behind an LSTM network in accuracy.

Preliminary hypothesis/conclusion: 
- rPGLU is able to capture the timing dynamics of the timeseries but does not retain this information well due to not having a proper hidden state vector.
- Potential solutions to try
  - Extend rPGLU with a hidden state memory by combining with LSTM, GLU, or some new architecture. 
  - Since the rPLGU seems to be able to process timing dynamics, a network on top could be added to process this information further. 

