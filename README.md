# Belief propagation in Ising models

A sum-product message-passing algorithm is implemented for an Ising model. The model is embedded in a generic graph created with networkx.

From the messages, we obtain the beliefs and the local magnetization and check that in a 2D lattice with periodic boundary condition we indeed find the Bethe critical point. The two-point beliefs, i.e. the joint probabilities over neighbouring variables, are used to estimate the two-point entropy cumulant.

The (local) two-point cumulant is the opposite of the (local) mutual information. The global mutual information is particularly important because it peaks at the critical point. Even though this algorithm does not estimate all the joint probabilities over all the possible pairs of variable, we see a peak at the critical point if we approximate the global mutual information using the global two-point cumulant.
