import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
from matplotlib.backends.backend_pdf import PdfPages


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
    noisy_signal = signal * np.random.normal(0, 0.01, size=n)
    d = 5

    alpha_j_1 = 1 / np.sqrt(n)  # alpha for 1st basic function of DCT (i.e. first column in dictionary matrix A)
    alpha_j_n = np.sqrt(2 / n)  # alpha for all other basic functions of DCT (i.e. every column except the first one in dictionary matrix A)

    A = np.empty((n, d))

    for j in range(1, d + 1):
        cos_part_1 = np.pi / n * (j - 1)
        for i in range(1, n + 1):
            cos_term = np.cos(cos_part_1 * (i - 1/2))
            if j == 1:
                A[i - 1, j - 1] = alpha_j_1 * cos_term
            else:
                A[i - 1, j - 1] = alpha_j_n * cos_term

    M = np.matmul(A.T, A)
    lambda_max = power_iteration(M)  # largest computed eigenvalue based on power-iteration algorithm

    """ End of your code
    """

    return fig


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