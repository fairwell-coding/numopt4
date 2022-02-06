import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
from matplotlib.backends.backend_pdf import PdfPages
from numpy.linalg import inv, norm
from numpy.random import set_state, SeedSequence, RandomState, MT19937

RANDOM_STATE = 42


def power_iteration(M: np.array, eps: float = 1e-8) -> float:
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

    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle('Task 1 - Signal denoising task', fontsize=16)

    ax[0, 0].set_title('a)')

    ax[0, 1].set_title('b)')

    ax[1, 0].set_title('c)')

    ax[1, 1].set_title('d)')

    """ Start of your code
    """

    # __perform_experiment(__perform_experiment_a(signal))
    # __perform_experiment(__perform_experiment_b(signal))
    __perform_experiment(__perform_experiment_c(signal))
    # __perform_experiment(__perform_experiment_d(signal))

    """ End of your code
    """

    return fig


def __perform_experiment(experiment):
    proj, selfmade, fw, _, _ = experiment
    diff_1 = fw[-1][0] - proj[-1][0]
    max_prj_fw = diff_1[np.argmax(abs(diff_1))]
    diff_2 = fw[-1][0] - selfmade[-1][0]
    max_custom_fw = diff_2[np.argmax(abs(diff_2))]
    diff_3 = selfmade[-1][0] - proj[-1][0]
    max_prj_custom = diff_3[np.argmax(abs(diff_3))]
    x_k_proj = proj[-1][0]
    x_k_selfmade = selfmade[-1][0]
    x_k_fw = fw[-1][0]

    print('x')


def __perform_experiment_a(signal):
    deviation = 0.01
    d = 15
    k = {'prj': 500, 'selfmade': 500, 'fw': 300}

    return __run_all_methods(d, k, signal, False, deviation)


def __perform_experiment_b(signal):
    deviation = 0.03
    d = 15
    k = {'prj': 300, 'selfmade': 300, 'fw': 400}

    return __run_all_methods(d, k, signal, False, deviation)


def __perform_experiment_c(signal):
    deviation = 0.01
    d = 100
    k = {'prj': 300, 'selfmade': 300, 'fw': 20}

    return __run_all_methods(d, k, signal, False, deviation)


def __perform_experiment_d(signal):
    deviation = 0.01
    d = 5
    k = {'prj': 300, 'selfmade': 300, 'fw': 20}

    return __run_all_methods(d, k, signal, False, deviation)


def __run_all_methods(d, k, signal, two_dimensional_input, deviation=0.0):
    n = signal.shape[0]

    if not two_dimensional_input:
        np.random.seed(RANDOM_STATE)
        noisy_signal = signal + np.random.normal(0, deviation, size=n)
        A, lambda_max = __calculate_lipschitz_constant(d, n)
    else:
        noisy_signal = signal
        A, lambda_max = __calculate_lipschitz_constant_2d(d, n)
        noisy_signal = np.ndarray.flatten(signal)  # flatten input image
        d **= 2
        k = {'prj': k, 'selfmade': k, 'fw': k}

    # Choose arbitrary initial value (i.e. x_0) for vector x within convex unit simplex
    x_k = np.zeros(d)
    x_k[0] = 1

    # Use verification example posted on forum by Christian Kopf
    # x_k = np.array([-0.4, 0.3, .5, 1.2])

    prj_gradient = __projected_gradient_method(x_k, A, lambda_max, noisy_signal, eps_step_size=1E-4, k=k['prj'])
    prj_gradient_custom = __selfmade_projection_method(x_k, A, d, lambda_max, noisy_signal, eps_step_size=1E-4, k=k['selfmade'])
    fw = __frank_wolfe_method(A, d, k['fw'], noisy_signal, x_k)
    if two_dimensional_input:
        fw_exact_line_search = __frank_wolfe_method(A, d, k['fw'], noisy_signal, x_k, True)
    else:
        fw_exact_line_search = None

    return prj_gradient, prj_gradient_custom, fw, fw_exact_line_search, A


def __project_on_unit_simplex(z):
    z_hat = -np.sort(-z)  # sort descending

    rho = __calculate_rho(z_hat)
    q = __calculate_q(rho, z_hat)
    x_k = __calculate_x_k(q, z)

    return x_k


def __calculate_x_k(q, z):
    x_k = np.empty(z.shape[0])
    for i in range(z.shape[0]):
        x_k[i] = np.maximum(z[i] + q, 0)
    return x_k


def __calculate_q(rho, z_hat):
    sum_rho_largest_coordinates = 0
    for i in range(rho):
        sum_rho_largest_coordinates += z_hat[i]
    q = 1 / rho * (1 - sum_rho_largest_coordinates)

    return q


def __calculate_rho(z_hat):
    rho = 0

    for i in range(z_hat.shape[0]):
        sum_of_bigger_coordinates = 0
        for j in range(i + 1):
            sum_of_bigger_coordinates += z_hat[j]
        r = z_hat[i] + 1 / (i + 1) * (1 - sum_of_bigger_coordinates)
        if r > 0:
            rho += 1

    return rho


def __projected_gradient_method(x_k, A, lambda_max, noisy_signal, eps_step_size, k):
    x_k_history = []  # history of projected x_k values over algorithm iterations (used to verify convergence)

    step_size = 2 / lambda_max - eps_step_size  # Choose step size t in (0, 2 / lipschitz_constant) minus an epsilon offset (so that convergence to global minimum is guaranteed when using numerical
    # operations)

    for i in range(k):
        z = __steepest_gradient_descent(A, noisy_signal, step_size, x_k)
        x_k = __project_on_unit_simplex(z)
        x_k_history.append((x_k, np.sum(x_k)))

    return x_k_history


def __frank_wolfe_method(A, d, k, noisy_signal, x_k, exact_line_search=False):
    x_k_history = []

    for i in range(k):
        gradient_obj_function = __calculate_gradient(A, noisy_signal, x_k)
        y_k = np.zeros(d)
        y_k[np.argmin(gradient_obj_function)] = 1  # e_i = p(x) = y_k = extremal point, element of linearized version of the cost function over the convex set

        if exact_line_search:  # tau_k = (b.T * A * (y_k - x_k) - x_k.T * A.T * A * (y_k - x_k)) / ((y_k.T - x_k.T) * A.T * A * (y_k - x_k)
            term_1 = np.matmul(A, (y_k - x_k))
            numerator = (np.matmul(noisy_signal.T, term_1) - np.matmul(x_k.T, np.matmul(A.T, term_1)))
            denominator = np.matmul(term_1.T, term_1)
            tau_k = numerator / denominator

            # ensure that tau_k lies between closed interval [0, 1]
            if tau_k < 0:
                tau_k = 0
            if tau_k > 1:
                tau_k = 1
        else:
            tau_k = 2 / (k + 1)  # tau_k = step size t_k

        x_k = (1 - tau_k) * x_k + tau_k * y_k
        x_k_history.append((x_k, np.sum(x_k)))

    return x_k_history


def __selfmade_projection_method(x_k, A, d, lambda_max, noisy_signal, eps_step_size, k):
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
    gradient_obj_function = __calculate_gradient(A, noisy_signal, x_k)
    z = x_k - step_size * gradient_obj_function  # z = steepest_descent = x_k - t * nabla_f(x)

    return z


def __calculate_gradient(A, noisy_signal, x_k):
    """ Calculated result by pen & paper: nabla_f(x) = A.T * (A * x - b)

    :param A: constructed dictionary matrix
    :param noisy_signal: Gaussian superimposed noisy signal vector b
    :param x_k: current iteration of solution vector x_k
    :return: gradient of current gradient descent (steepest descent) = nabla_f(x)
    """

    return np.matmul(A.T, np.matmul(A, x_k) - noisy_signal)


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


def __flatten_first_two_dimensions(n: np.ndarray):
    """ Takes 3d tensor as input, flattens the first two dimensions and returns the resulting 2d tensor.

    :param n: 3d input tensor
    :return: transformed 2d output tensor (first two dimensions are C-order flattened
    """

    n_transformed = np.empty((n.shape[0] * n.shape[1], n.shape[2]))

    for i in range(n.shape[2]):
        n_transformed[:, i] = n[:, :, i].flatten()

    return n_transformed


def __calculate_lipschitz_constant_2d(d, n):
    # Define often needed values
    alpha_l_1 = 1 / np.sqrt(n)
    alpha_l_d = np.sqrt(2 / n)  # alpha for all other basic functions of DCT (i.e. every column except the first d ones in dictionary matrix A), i.e. d^2 - d

    alpha_l = np.full(d, alpha_l_d)
    alpha_l[0] = alpha_l_1

    alpha_l_m = np.matmul(alpha_l.reshape((d, 1)), alpha_l.reshape((1, d))).flatten()

    # Define empty dictionary matrix for DCT
    A = np.empty((n, n, d ** 2))

    # Calculate dictionary matrix
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            cos_lm_vector = np.empty((d, d))
            for l in range(1, d + 1):
                for m in range(1, d + 1):
                    cos_1 = np.cos(np.pi / n * (l - 1) * (i - 1 / 2))
                    cos_2 = np.cos(np.pi / n * (m - 1) * (j - 1 / 2))
                    cos_lm_vector[l - 1, m - 1] = cos_1 * cos_2
            A[i - 1, j - 1] = cos_lm_vector.reshape(d ** 2)

    A = __flatten_first_two_dimensions(A) * alpha_l_m  # calculated dictionary matrix for DCT: shape = (n**2, d**2)

    M = np.matmul(A.T, A)  # calculate hessian
    lambda_max = power_iteration(M)  # largest computed eigenvalue based on power-iteration algorithm = Lipschitz constant

    return A, lambda_max


def __calculate_lipschitz_constant(d, n):
    alpha_j_1 = __calculate_alpha_j_1(n)
    alpha_j_n = np.sqrt(2 / n)  # alpha for all other basic functions of DCT (i.e. every column except the first one in dictionary matrix A)

    A = np.empty((n, d))  # dictionary matrix for DCT

    for j in range(1, d + 1):
        cos_part_1 = np.pi / n * (j - 1)
        for i in range(1, n + 1):
            cos_term = np.cos(cos_part_1 * (i - 1 / 2))
            if j == 1:
                A[i - 1, j - 1] = alpha_j_1 * cos_term
            else:
                A[i - 1, j - 1] = alpha_j_n * cos_term

    M = np.matmul(A.T, A)
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

    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    fig.suptitle('Task 2 - Image Representation', fontsize=16)
    ax = [None, None, None, None, None, None]

    # g = fig.add_gridspec(9, 9)
    # ax[0] = fig.add_subplot(g[1:4, 0:3])
    # ax[1] = fig.add_subplot(g[1:4, 3:6])
    # ax[2] = fig.add_subplot(g[1:4, 6:])
    # ax[3] = fig.add_subplot(g[4:, :])

    g = fig.add_gridspec(12, 9)
    ax[0] = fig.add_subplot(g[1:4, 0:3])
    ax[1] = fig.add_subplot(g[1:4, 3:6])
    ax[2] = fig.add_subplot(g[1:4, 6:9])
    ax[3] = fig.add_subplot(g[4:7, 3:6])
    ax[4] = fig.add_subplot(g[4:7, 6:9])
    ax[5] = fig.add_subplot(g[7:, :])

    for ax_ in ax[:-1]:
        ax_.set_aspect('equal')
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    ax[0].set_title('Ground Truth image')
    ax[1].set_title('Proj. GD')
    ax[2].set_title('Custom Proj. GD')
    ax[3].set_title('FW: predefined diminishing step size')
    ax[4].set_title('FW: exact line search')

    """ Start of your code
    """

    d = 2
    k = 1500
    prj_gradient, prj_gradient_custom, fw, fw_exact_line_search, A = __run_all_methods(d=d, k=k, signal=img, two_dimensional_input=True)

    # Ground truth image
    ax[0].imshow(img)

    # Projected gradient method
    prj_gradient_img = np.matmul(A, prj_gradient[-1][0]).reshape((img.shape[0], img.shape[1]))
    ax[1].imshow(prj_gradient_img)

    # Custom projected gradient method
    prj_gradient_custom_img = np.matmul(A, prj_gradient_custom[-1][0]).reshape((img.shape[0], img.shape[1]))
    ax[2].imshow(prj_gradient_custom_img)

    # Frank-Wolfe using predefined diminishing step size
    fw_img = np.matmul(A, fw[-1][0]).reshape((img.shape[0], img.shape[1]))
    ax[3].imshow(fw_img)

    # Frank-Wolfe using exact line search
    fw_exact_line_search_img = np.matmul(A, fw_exact_line_search[-1][0]).reshape((img.shape[0], img.shape[1]))
    ax[4].imshow(fw_exact_line_search_img)

    obj_fun_proj = __calculate_progression_of_objective_function(A, img, prj_gradient)
    obj_fun_proj_custom = __calculate_progression_of_objective_function(A, img, prj_gradient_custom)
    obj_fun_fw = __calculate_progression_of_objective_function(A, img, fw)
    obj_fun_fw_exact_line_search = __calculate_progression_of_objective_function(A, img, fw_exact_line_search)

    iterations = np.arange(1, k + 1)  # k=1500 iterations
    obj_fun_proj_handle, = ax[5].semilogy(iterations, obj_fun_proj, color="forestgreen", label="Projected gradient method")
    obj_fun_proj_custom_handle, = ax[5].semilogy(iterations, obj_fun_proj_custom, color="darkred", label="Custom projected gradient method (our derived formula)")
    obj_fun_fw_handle, = ax[5].semilogy(iterations, obj_fun_fw, color="mediumblue", label="Frank Wolfe: predefined diminishing step size")
    obj_fun_fw_exact_line_search_handle, = ax[5].semilogy(iterations, obj_fun_fw_exact_line_search, color="darkorange", label="Frank Wolfe: exact line search")
    ax[5].legend(handles=[obj_fun_proj_handle, obj_fun_proj_custom_handle, obj_fun_fw_handle, obj_fun_fw_exact_line_search_handle])

    """ End of your code
    """

    return fig


def __calculate_progression_of_objective_function(A, img, method_history):
    obj_function_values = []
    for hist in method_history:
        x_k = hist[0]
        obj_function_values.append(1 / 2 * norm(np.matmul(A, x_k) - np.ndarray.flatten(img), 2))

    return obj_function_values

if __name__ == "__main__":
    args = []
    with np.load('data.npz') as data:
        # args.append(data['sig'])
        args.append(data['yoshi'])

    pdf = PdfPages('figures.pdf')

    # for task, arg in zip([task1, task2], args):
    for task, arg in zip([task2], args):
        retval = task(arg)
        fig = retval[0] if type(retval) is tuple else retval
        pdf.savefig(fig)
    pdf.close()
