import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
from matplotlib.backends.backend_pdf import PdfPages
from numpy.linalg import inv


def power_iteration(M: np.array, eps:float=1e-8) -> float:

    assert len(M.shape) == 2, 'M must be matrix with 2 dimensions'
    assert M.shape[0] == M.shape[1], 'M must be a square matrix'

    d = M.shape[1]

    v = np.ones(d) / np.sqrt(d)
    ev = v @ M @ v

    while True:
        Mv = M @ v
        v_new = Mv / (np.linalg.norm(Mv))

        ev_new = v_new @ M @ v_new
        if np.abs(ev - ev_new) < eps:
            break

        v = v_new
        ev = ev_new

    # Return the largest eigenvalue
    return ev_new


def task1(signal):
    """ Signal Denoising

        Requirements for the plots:
            -ax[0,0] - Results for low noise and K  =15
            -ax[0,1] - Results for high noise and K=15
            -ax[1,0] - Results for low noise and K=100
            -ax[1,1] - Results for low noise and K=5

    """

    fig, ax = plt.subplots(2, 2, figsize=(16,8))
    fig.suptitle('Task 1 - Signal denoising task', fontsize=16)

    ax[0,0].set_title('a)')

    ax[0,1].set_title('b)')

    ax[1,0].set_title('c)')

    ax[1,1].set_title('d)')

    """ Start of your code
    """

    # Create dictionary matrix for DCT (discrete cosine transform)
    n = signal.shape[0]

    history_a = __perform_experiment_a(n, signal)
    history_b = __perform_experiment_b(n, signal)
    history_c = __perform_experiment_c(n, signal)
    history_d = __perform_experiment_d(n, signal)

    """ End of your code
    """

    return fig


def __perform_experiment_a(n, signal):
    deviation = 0.01
    d = 15
    k = 100

    return __run_both_algorithms(d, deviation, k, n, signal)


def __perform_experiment_b(n, signal):
    deviation = 0.03
    d = 15
    k = 100

    return __run_both_algorithms(d, deviation, k, n, signal)


def __perform_experiment_c(n, signal):
    deviation = 0.01
    d = 100
    k = 100

    return __run_both_algorithms(d, deviation, k, n, signal)


def __perform_experiment_d(n, signal):
    deviation = 0.01
    d = 5
    k = 100

    return __run_both_algorithms(d, deviation, k, n, signal)


def __run_both_algorithms(d, deviation, k, n, signal):
    noisy_signal = signal * np.random.normal(0, deviation, size=n)
    A, lambda_max = __calculate_lipschitz_constant(d, n)
    training_history = __projected_gradient_method(A, d, lambda_max, noisy_signal, eps_step_size=1E-4, k=k)
    # x_k_1 = training_history[-1]

    return training_history


def __projected_gradient_method(A, d, lambda_max, noisy_signal, eps_step_size, k):
    x_k = np.zeros(d)  # Choose arbitrary initial value (i.e. x_0) for vector x within convex unit simplex
    x_k[0] = 1

    x_k_history = []  # history of projected x_k values over algorithm iterations (used to verify convergence)

    step_size = 2 / lambda_max - eps_step_size  # Choose step size t in (0, 2 / lipschitz_constant) minus an epsilon offset (so that convergence to global minimum is guaranteed when using numerical
    # operations)

    direction_vector_matrix, p = __define_hyperplane(d)

    for i in range(k):
        z = __steepest_gradient_descent(A, noisy_signal, step_size, x_k)
        projection_on_hyperplane = __projection_on_unconstrained_hyperplane(d, direction_vector_matrix, p, z)
        x_k = __projection_from_hyperplane_to_unit_simplex(d, projection_on_hyperplane)
        x_k_history.append((x_k, np.sum(x_k)))

    return x_k_history


def __projection_on_unconstrained_hyperplane(d, direction_vector_matrix, p, z):
    # Define normal vector of hyperplane (not normalized since this is irrelevant to calculate intersection)
    normal_vector = np.full(d, 1)

    # Calculate distance of intersection of hyperplane with normal vector
    D = np.concatenate((direction_vector_matrix, -1 * normal_vector.reshape((normal_vector.shape[0], 1))), axis=1)  # matrix D combines direction vector matrix and adds negative normal vector as
    # last column
    scaling_vector = np.matmul(inv(D), z - p)
    projection_on_hyperplane = z + scaling_vector[-1] * normal_vector

    return projection_on_hyperplane


def __steepest_gradient_descent(A, noisy_signal, step_size, x_k):
    gradient_obj_function = np.matmul(A.T, np.matmul(A, x_k) - noisy_signal)  # Calculated result by pen & paper: nabla_f(x) = A.T * (A * x - b)
    z = x_k - step_size * gradient_obj_function  # z = steepest_descent = x_k - t * nabla_f(x)

    return z


def __define_hyperplane(d):
    """ Define hyperplane by using a point p on the hyperplane and d-1 direction vectors

    :param d:
    :return:
    """

    p = np.zeros(d)
    p[0] = 1

    direction_vector_matrix = np.zeros((d, d - 1))  # d-1 direction vectors span the hyperplane (subspace of dim=d-1); each vector exists in vector space of dim=d
    for j in range(d - 1):
        direction_vector_matrix[0, j] = 1
        direction_vector_matrix[j + 1, j] = -1

    return direction_vector_matrix, p


def __projection_from_hyperplane_to_unit_simplex(d, projection_on_hyperplane):
    """ Projection from d-1 dim hyperplane (i.e. some coordinates may still be negative and thus violate unit simplex constraint) -> d-1 dim unit simplex space

    :param d: dimension of enclosing vector space
    :param projection_on_hyperplane: projected solution on hyperplane
    :return: projected x* (equivalent to x_k_1 notation) for current iteration
    """

    binary_projection_mask = np.full(d, 1)

    while np.sum(np.array(projection_on_hyperplane) < 0) > 0:  # negative values exist
        for j in range(d):
            if projection_on_hyperplane[j] > 0:
                continue

            binary_projection_mask[j] = 0
            compensation_factor = - projection_on_hyperplane[j]
            projection_on_hyperplane -= compensation_factor / np.sum(binary_projection_mask) * binary_projection_mask
            projection_on_hyperplane[j] = 0

    x_k_1 = projection_on_hyperplane

    return x_k_1


def __calculate_lipschitz_constant(d, n):
    alpha_j_1 = __calculate_alpha_j_1(n)
    alpha_j_n = np.sqrt(2 / n)  # alpha for all other basic functions of DCT (i.e. every column except the first one in dictionary matrix A)

    A = np.empty((n, d))  # dictionary matrix for DCT

    for j in range(1, d + 1):
        M = np.matmul(A.T, A)
        cos_part_1 = np.pi / n * (j - 1)
        for i in range(1, n + 1):
            cos_term = np.cos(cos_part_1 * (i - 1 / 2))
            if j == 1:
                A[i - 1, j - 1] = alpha_j_1 * cos_term
            else:
                A[i - 1, j - 1] = alpha_j_n * cos_term

    lambda_max = power_iteration(M)  # largest computed eigenvalue based on power-iteration algorithm = Lipschitz constant

    return A, lambda_max


def __calculate_alpha_j_1(n):
    alpha_j_1 = 1 / np.sqrt(n)  # alpha for 1st basic function of DCT (i.e. first column in dictionary matrix A)
    return alpha_j_1


def task2(img):

    """ Image Representation

        Requirements for the plots:
            - ax[0] The ground truth image
            - ax[1] Reconstructed image using proj. GD
            - ax[2] Reconstructed image using Frank-Wolfe algorithm
            - ax[3] Semilogarithmic plot comparing the energies over
                    iterations for both methods
    """

    fig = plt.figure(figsize=(11,9), constrained_layout=True)
    fig.suptitle('Task 2 - Image Representation', fontsize=16)
    ax = [None, None, None, None]
    g = fig.add_gridspec(9, 9)
    ax[0] = fig.add_subplot(g[1:4:, 0:3])
    ax[1] = fig.add_subplot(g[1:4:, 3:6])
    ax[2] = fig.add_subplot(g[1:4:, 6:])
    ax[3] = fig.add_subplot(g[4:, :])

    for ax_ in ax[:-1]:
        ax_.set_aspect('equal')
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)
    
    ax[0].set_title('GT')
    ax[1].set_title('Proj. GD')
    ax[2].set_title('Frank-Wolfe')
    
    """ Start of your code
    """

    """ End of your code
    """

    return fig

    
if __name__ == "__main__":
    args = []
    with np.load('data.npz') as data:
        args.append(data['sig'])
        args.append(data['yoshi'])
    
    pdf = PdfPages('figures.pdf')

    for task, arg in zip([task1, task2], args):
        retval = task(arg)
        fig = retval[0] if type(retval) is tuple else retval
        pdf.savefig(fig)
    pdf.close()