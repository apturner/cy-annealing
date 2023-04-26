import itertools
from dimod import BinaryPolynomial, make_quadratic, BINARY
from dwave.samplers import SimulatedAnnealingSampler
from dimod import ExactSolver, ExactPolySolver
from ast import literal_eval
from functools import reduce

# tensors, with each distinct entry only specified once
dim = 2

int_tens_1 = {(1, 1, 1): 0, (1, 1, 2): 1, (1, 2, 2): -3, (2, 2, 2): 9}

# correctly found with res 1, det_strength 10.0**2, reduction strength 10.0**5, n_samples = 10**3, n_sweeps 10**3, n_per_beta 10**2, beta_bounds [0.1, 10.0], beta_type = "geometric"
int_tens_1_a1 = {(1, 1, 1): 3, (1, 1, 2): 4, (1, 2, 2): 6, (2, 2, 2): 9}
int_tens_1_a1_rot = [1, 1, 0, 1]

int_tens_1_a2 = {(1, 1, 1): -39, (1, 1, 2): -46, (1, 2, 2): -54, (2, 2, 2): -63}
int_tens_1_a2_rot = [2, -1, 3, -1]

int_tens_1_a3 = {(1, 1, 1): 42, (1, 1, 2): 25, (1, 2, 2): 15, (2, 2, 2): 9}
int_tens_1_a3_rot = [5, 2, 3, 1]

int_tens_2 = {(1, 1, 1): 0, (1, 1, 2): 2, (1, 2, 2): 2, (2, 2, 2): 2}

# correctly found with res 1, det_strength 10.0**2, reduction strength 10.0**5, n_samples = 10**3, n_sweeps 10**3, n_per_beta 10**2, beta_bounds [0.1, 10.0], beta_type = "geometric"
int_tens_2_a1 = {(1, 1, 1): -2, (1, 1, 2): 0, (1, 2, 2): 0, (2, 2, 2): 2}
int_tens_2_a1_rot = [1, -1, 0, 1]

# tensors to compare
int_tens_in_1 = int_tens_1
int_tens_in_2 = int_tens_1_a1
correct_rot = int_tens_1_a1_rot

# parameters for defining polynomial
res = 2  # range of values is -2^res to 2^res - 1
det_strength = 10.0**6
reduction_strength = 10.0**5

# parameters for annealing
n_samples = 10**4
n_sweeps = 10**4
n_per_beta = n_sweeps // 10
beta_bounds = [0.1, 10.0]
beta_type = "geometric"

# polynomial terms
sextics = {
    ((i, a, r1), (j, b, r2), (k, c, r3), (i, x, r4), (j, y, r5), (k, z, r6)): (-1)
    ** ([r1, r2, r3, r4, r5, r6].count(res))
    * 2 ** (r1 + r2 + r3 + r4 + r5 + r6)
    * int_tens_in_1[tuple(sorted([a, b, c]))]
    * int_tens_in_1[tuple(sorted([x, y, z]))]
    for (i, j, k, a, b, c, x, y, z) in itertools.product(range(1, dim + 1), repeat=9)
    for (r1, r2, r3, r4, r5, r6) in itertools.product(range(res + 1), repeat=6)
}
cubics = {
    ((i, a, r1), (j, b, r2), (k, c, r3)): -2
    * (-1) ** ([r1, r2, r3].count(res))
    * 2 ** (r1 + r2 + r3)
    * int_tens_in_1[tuple(sorted([a, b, c]))]
    * int_tens_in_2[tuple(sorted([i, j, k]))]
    for (i, j, k, a, b, c) in itertools.product(range(1, dim + 1), repeat=6)
    for (r1, r2, r3) in itertools.product(range(res + 1), repeat=3)
}
det = (
    {
        ((1, 1, r1), (1, 1, r2), (2, 2, r3), (2, 2, r4)): det_strength
        * (-1) ** ([r1, r2, r3, r4].count(res))
        * 2 ** (r1 + r2 + r3 + r4)
        for (r1, r2, r3, r4) in itertools.product(range(res + 1), repeat=4)
    }
    | {
        ((1, 2, r1), (1, 2, r2), (2, 1, r3), (2, 1, r4)): det_strength
        * (-1) ** ([r1, r2, r3, r4].count(res))
        * 2 ** (r1 + r2 + r3 + r4)
        for (r1, r2, r3, r4) in itertools.product(range(res + 1), repeat=4)
    }
    | {
        ((1, 1, r1), (2, 2, r2), (1, 2, r3), (2, 1, r4)): det_strength
        * (-2)
        * (-1) ** ([r1, r2, r3, r4].count(res))
        * 2 ** (r1 + r2 + r3 + r4)
        for (r1, r2, r3, r4) in itertools.product(range(res + 1), repeat=4)
    }
    | {
        ((1, 1, r1), (2, 2, r2)): det_strength
        * (-2)
        * (-1) ** ([r1, r2].count(res))
        * 2 ** (r1 + r2)
        for (r1, r2) in itertools.product(range(res + 1), repeat=2)
    }
    | {
        ((1, 2, r1), (2, 1, r2)): det_strength
        * 2
        * (-1) ** ([r1, r2].count(res))
        * 2 ** (r1 + r2)
        for (r1, r2) in itertools.product(range(res + 1), repeat=2)
    }
)
poly_terms = sextics | cubics | det
# poly_terms = det

# create and reduce polynomial
poly = BinaryPolynomial(poly_terms, BINARY)
quad = make_quadratic(poly, reduction_strength, BINARY)
quad.normalize()


class Printer:
    def __init__(self):
        self.counter = 0
        self.percent = 0

    def __call__(self):
        self.counter += 1
        if self.counter % (n_samples // 100) == 0:
            self.percent += 1
            print(self.percent, "%")
        return False


# sample
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(
    quad,
    num_reads=n_samples,
    num_sweeps=n_sweeps,
    num_sweeps_per_beta=n_per_beta,
    beta_range=beta_bounds,
    beta_schedule_type=beta_type,
    interrupt_function=Printer(),
)

# aggregate samples and pick best solution
aggregate = sampleset.aggregate()
vars = aggregate.variables
first_sol = aggregate.first.sample
result = [
    sum(
        (-1 if r == res else 1) * 2**r * first_sol[(1, 1, r)] for r in range(res + 1)
    ),
    sum(
        (-1 if r == res else 1) * 2**r * first_sol[(1, 2, r)] for r in range(res + 1)
    ),
    sum(
        (-1 if r == res else 1) * 2**r * first_sol[(2, 1, r)] for r in range(res + 1)
    ),
    sum(
        (-1 if r == res else 1) * 2**r * first_sol[(2, 2, r)] for r in range(res + 1)
    ),
]
print("Correct answer found: ", result == correct_rot)
print(
    "Determinant constraint satisfied: ",
    result[0] * result[3] - result[1] * result[2] == 1,
)
reduction_constraints_satisfied = True
for vari in vars:
    if type(vari) == str:
        var_prod = reduce(
            lambda prod, var_name: prod * first_sol[literal_eval(var_name)],
            vari.split("*"),
            1,
        )
        reduction_constraints_satisfied == reduction_constraints_satisfied & var_prod == first_sol[
            vari
        ]
print("Reduction constraints satisfied: ", reduction_constraints_satisfied, "\n")

# solution details
print("Correct answer: ", correct_rot)
print("Result: ", result)
print("Determinant: ", result[0] * result[3] - result[1] * result[2])
print("Energy: ", aggregate.first.energy)

# exact solution
# exact_sampler = ExactSolver()
# exact_sample = exact_sampler.sample(quad)
# first_exact_sol = exact_sample.first.sample
# exact_result = [
#     sum(
#         (-1 if r == res else 1) * 2**r * first_exact_sol[(1, 1, r)]
#         for r in range(res + 1)
#     ),
#     sum(
#         (-1 if r == res else 1) * 2**r * first_exact_sol[(1, 2, r)]
#         for r in range(res + 1)
#     ),
#     sum(
#         (-1 if r == res else 1) * 2**r * first_exact_sol[(2, 1, r)]
#         for r in range(res + 1)
#     ),
#     sum(
#         (-1 if r == res else 1) * 2**r * first_exact_sol[(2, 2, r)]
#         for r in range(res + 1)
#     ),
# ]
# # print(exact_sample)
# print("Exact Result: ", exact_result)
# print(
#     "Exact Determinant: ",
#     exact_result[0] * exact_result[3] - exact_result[1] * exact_result[2],
# )
# print("Exact energy: ", exact_sample.first.energy)

# exact polynomial solution
# exact_poly_sampler = ExactPolySolver()
# exact_poly_sample = exact_poly_sampler.sample_poly(poly)
# first_exact_poly_sol = exact_poly_sample.first.sample
# exact_poly_result = [
#     [
#         sum(
#             (-1 if r == res else 1) * 2**r * first_exact_poly_sol[(1, 1, r)]
#             for r in range(res + 1)
#         ),
#         sum(
#             (-1 if r == res else 1) * 2**r * first_exact_poly_sol[(1, 2, r)]
#             for r in range(res + 1)
#         ),
#     ],
#     [
#         sum(
#             (-1 if r == res else 1) * 2**r * first_exact_poly_sol[(2, 1, r)]
#             for r in range(res + 1)
#         ),
#         sum(
#             (-1 if r == res else 1) * 2**r * first_exact_poly_sol[(2, 2, r)]
#             for r in range(res + 1)
#         ),
#     ],
# ]
# print(exact_poly_result)
# print(
#     exact_poly_result[0][0] * exact_poly_result[1][1]
#     - exact_poly_result[0][1] * exact_poly_result[1][0]
# )
# print(exact_poly_sample.first.energy)

print("\a")
