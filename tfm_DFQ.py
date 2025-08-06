# tfm_DFQ
# ### Beginning of the code implementation for the Double Factorization Qubitization method
# 
# Based on the article https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.033055, which describes the second factorization qubitization method.
# 

# %%
import math
import sympy
import numpy as np
from scipy.special import binom
import scipy

# %%
class DoubleFactorization:
    def __init__(self, tools):

        self.tools = tools

    # Qubitization (PhysRevResearch.3.033055)
        """
        We are going to calculate the cost of qubitization based on the number of Toffoli gates required.
        The cost is estimated based on the number of eigenvectors in the double factorization, the number of orbitals,
        the target precision, and the ancilla control parameter.
        """
    def estimate_beta(self, N, epsilon):
        """
        Calculation of beta as base-2 logarithm.
        Quantifies how many ancillary qubits are needed to represent indices or errors with sufficient precision.
        Expression that appears in the text before equation (20).
        """
        #return math.ceil(5.652 + np.log2(N / epsilon))
        return math.ceil(5.652 + math.log2(N / epsilon))

    def toffoli_gate_cost(self, M, N, epsilon, alpha_df, lambda_=2):
        """
        Estimate the number of Toffoli gates required for qubitization with a doubly-factorized representation.

        Parameters
        ----------
        M (int):  Number of eigenvectors in the double factorization.
        N (int): Number of orbitals.
        epsilon (float): Target precision (in Hartree).
        lambda_ (float):Ancilla control parameter (typical value: 1–5).
        alpha_df (float): Normalization constant.
        epsilon (float): Target precision (in Hartree).
        c_w (int): Cost of the walk operator (typically proportional to the number of terms L).

        Returns
        -------
        cW (int): Cost in number of Toffoli gates.
        """
        # Approximation of equation (19):
        return (2 * M/(1 + lambda_) + 2 * lambda_ * N * self.estimate_beta(N, epsilon) + 8 * N * self.estimate_beta(N, epsilon) + 4 * N) * np.pi * alpha_df / (2*epsilon) 
    
    
    def double_factorization_method(self, errors, p_fail, lambda_1, lambda_2, rank_1, rank_2, N, L):
        """
        Método de cálculo del coste en puertas Toffoli para el algoritmo de Doble Factorización y Qubitización.

        Parameters
        ----------
        errors : list of float
            Lista de errores [epsilon_PEA, epsilon_HS, epsilon_S].
        p_fail : float
            Probabilidad de fallo total.
        lambda_1 : float
            Constante de normalización (primer nivel).
        lambda_2 : float
            Constante de normalización (segundo nivel).
        rank_1 : int
            Número de términos del primer nivel de factorización.
        rank_2 : int
            Número de términos del segundo nivel de factorización.
        N : int
            Número de electrones (o número de orbitales activos).
        L : int
            Número total de orbitales.

        Returns
        -------
        float
            Coste estimado en número de puertas Toffoli.
        """
        if not self.has_data:
            print("Skipping double factorization: no geometry loaded.")
            return
        
        # Desempaquetamos errores (solo se usa epsilon_PEA)
        epsilon_pea = errors[0]

        # Alpha_df = lambda_1 * lambda_2
        alpha_df = lambda_1 * lambda_2
        M = rank_1 * rank_2

        return self.toffoli_gate_cost(M, N, epsilon_pea, alpha_df)


# %%
#Example based on Table I of the article (structure VIII)
M = 39088       # number of eigenvectors
N = 65          # number of orbitals
epsilon = 1e-3  # precision of 1 mHartree
lambda_ = 2   # ancillas
alpha_df= 425.7  # normalization constant 

qb = DoubleFactorization(tools=np)
cost = qb.toffoli_gate_cost(M, N, epsilon, alpha_df, lambda_)
#print(f"Estimated Beta: {qb.estimate_beta(N, epsilon)}")
#print(f"Estimated cost in Toffoli gates: {cost / 1e10:.2f} × 10¹⁰")

# cd TFermion

#python3 main.py water double_factorization


