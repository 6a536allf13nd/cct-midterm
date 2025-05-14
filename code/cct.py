import pandas as pd
import numpy as np

# generate w/ ChatGPT

def load_plant_knowledge_data(path="data/plant_knowledge.csv"):
    df = pd.read_csv(path)

    # Drop any non-numeric columns (like question IDs)
    df = df.select_dtypes(include='number')

    data = df.to_numpy().astype(int)
    print("Data loaded:", data.shape)
    return data


import pymc as pm 
import arviz as az

def run_cct_model(data):
    N, M = data.shape  # N = number of informants, M = number of questions

    with pm.Model() as model:
        # Prior for informant competence: Uniform between 0.5 and 1
        D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)

        # Prior for consensus answers: Bernoulli(0.5)
        Z = pm.Bernoulli("Z", p=0.5, shape=M)

        # Reshape D for broadcasting
        D_reshaped = D[:, None]  # shape (N, 1)

        # Compute pij for each informant-question pair
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)

        # Likelihood
        X = pm.Bernoulli("X", p=p, observed=data)

        # Sampling
        trace = pm.sample(2000, chains=4, tune=1000, return_inferencedata=True)

    return trace


if __name__ == "__main__":
    data = load_plant_knowledge_data()
    print("Data loaded:", data.shape)

    trace = run_cct_model(data)
    az.to_netcdf(trace, "trace.nc")



# Load the trace from disk
trace = az.from_netcdf("trace.nc")

# Summarize the posterior distributions (includes R-hat, ESS, etc.)
summary = az.summary(trace, var_names=["D", "Z"])
print(summary)

# Plot posterior distributions for informant competence
az.plot_posterior(trace, var_names=["D"])

# Estimate mean competence per informant
print(trace.posterior["D"].mean(dim=["chain", "draw"]))

# Most & Least Competent Informants
competence = summary.loc[summary.index.str.startswith("D"), "mean"]
most_competent = competence.idxmax(), competence.max()
least_competent = competence.idxmin(), competence.min()

# Plot posterior distributions for consensus answers
az.plot_posterior(trace, var_names=["Z"])

# Estimate consensus answer key
consensus_probs = trace.posterior["Z"].mean(dim=["chain", "draw"])
consensus_key = (consensus_probs > 0.5).astype(int)
print("Consensus answer key:", consensus_key.values)

# Check convergence diagnostics
az.plot_trace(trace, var_names=["D", "Z"])

# Compare with Majority Vote
data = load_plant_knowledge_data()
majority_vote = (np.mean(data, axis=0) > 0.5).astype(int)
print("Majority vote:", majority_vote)

