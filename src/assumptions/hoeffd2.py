import numpy as np
import scipy.stats
from scipy.stats import rankdata


def hoeffd2(X, Y):
    # New dataset of 10 data points representing pairs of (height_in_inches, weight_in_pounds), with the last 3 being ties:
    # The 'average' ranking method assigns the average of the ranks that would have been assigned to all the tied values.
    R = rankdata(X, method='average')  # Rank of heights
    S = rankdata(Y, method='average')  # Rank of weights
    # Print the ranks for visualization. Ties are evident in the last three entries of both rankings.
    #print(f"Ranks of Heights (X): {R}")
    #print(f"Ranks of Weights (Y): {S}")
    N = len(X)  # Total number of data points
    # Q is an array that will hold a special sum for each data point, which is crucial for Hoeffding's D computation.
    Q = np.zeros(N)
    # Loop through each data point to calculate its Q value.
    for i in range(N):
        # For each data point 'i', count how many points have both a lower height and weight rank (concordant pairs).
        Q[i] = 1 + sum(np.logical_and(R < R[i], S < S[i]))

        # Adjust Q[i] for ties: when both ranks are equal, it contributes partially (1/4) to the Q[i] value.
        # The "- 1" accounts for not including the point itself in its own comparison.
        Q[i] += (1 / 4) * (sum(np.logical_and(R == R[i], S == S[i])) - 1)

        # When only the height rank is tied but the weight rank is lower, it contributes half (1/2) to the Q[i] value.
        Q[i] += (1 / 2) * sum(np.logical_and(R == R[i], S < S[i]))

        # Similarly, when only the weight rank is tied but the height rank is lower, it also contributes half (1/2).
        Q[i] += (1 / 2) * sum(np.logical_and(R < R[i], S == S[i]))
    # Print the Q values for each data point, indicating the weighted count of points considered "lower" or "equal".
    #print(f"Q values: {Q}")
    # Calculate intermediate sums required for Hoeffding's D formula:
    # D1: This sum leverages the Q values calculated earlier. Each Q value encapsulates information about how
    # a data point's ranks relate to others in both sequences, including concordance and adjustments for ties.
    # The term (Q - 1) * (Q - 2) for each data point quantifies the extent to which the ranks of this point
    # are concordant with others, adjusted for the expected concordance under independence.
    # Summing these terms across all data points (D1) aggregates this concordance information for the entire dataset.
    D1 = sum((Q - 1) * (Q - 2))
    # D2: This sum involves products of rank differences for each sequence, adjusted for ties. The term
    # (R - 1) * (R - 2) * (S - 1) * (S - 2) for each data point captures the interaction between the rank variances
    # within each sequence, providing a measure of how the joint rank distribution diverges from what would
    # be expected under independence due to the variability in ranks alone, without considering their pairing.
    # Summing these products across all data points (D2) gives a global assessment of this divergence.
    D2 = sum((R - 1) * (R - 2) * (S - 1) * (S - 2))
    # D3: This sum represents an interaction term that combines the insights from Q values with rank differences.
    # The term (R - 2) * (S - 2) * (Q - 1) for each data point considers the rank variances alongside the Q value,
    # capturing how individual data points' rank concordance/discordance contributes to the overall dependency measure,
    # adjusted for the expected values under independence. Summing these terms (D3) integrates these individual
    # contributions into a comprehensive interaction term for the dataset.
    D3 = sum((R - 2) * (S - 2) * (Q - 1))
    # The final computation of Hoeffding's D integrates D1, D2, and D3, along with normalization factors
    # that account for the sample size (N). The normalization ensures that Hoeffding's D is scaled appropriately,
    # allowing for meaningful comparison across datasets of different sizes. The formula incorporates these sums
    # and normalization factors in a way that balances the contributions of concordance, discordance, and rank variances,
    # resulting in a statistic that robustly measures the degree of association between the two sequences.
    D = 30 * ((N - 2) * (N - 3) * D1 + D2 - 2 * (N - 2) * D3) / (N * (N - 1) * (N - 2) * (N - 3) * (N - 4))
    # Return the computed Hoeffding's D value.
    return D
