# cct-midterm
Cultural Consensus Theory Report

I implemented a Bayesian Cultural Consensus Theory model using PyMC to analyze binary response data from 10 informants across 20 plant knowledge questions. The model estimated each informant’s competence (i.e., their probability of answering correctly) and a consensus answer key representing culturally correct responses. I used a Beta(2, 2) prior for informant competence to reflect moderate prior uncertainty, and a Bernoulli(0.5) prior for each item in the consensus key. The model showed good convergence (R-hat values = 1.00), and posterior estimates were stable across chains.

The consensus answer key from the model was:
[0 1 1 0 1 1 1 1 1 1 1 0 0 1 0 0 0 0 0 1]
The majority vote answer key was:
[0 0 1 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1]

These two differed on 5 questions. These differences occur because the CCT model accounts for varying competence levels among informants. These differences highlight the influence of competence-weighting in the CCT model: rather than treating each informant equally, the model estimates individual competence levels and gives more weight to answers from highly competent individuals. As a result, the consensus key is less biased by majority error and more reflective of informants who consistently demonstrate accurate knowledge. 

