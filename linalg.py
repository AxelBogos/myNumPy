from numpy import *

def square(a):
    '''
      This function tests if a matrix is square. It returns True
            if a represents a square matrix.
    '''

    return shape(a)[0] == shape(a)[1]

def swap(a, i, j):
    if len(shape(a)) == 1:
        a[i], a[j] = a[j], a[i]  # unpacking
    else:
        a[[i, j], :] = a[[j, i], :]

def gauss_elimination(a, b, verbose=False):
    n, m = shape(a)
    n2 = shape(b)[0]
    assert (n == n2)
    for k in range(n - 1):
        for i in range(k + 1, n):
            assert (a[k, k] != 0)
            if (a[i, k] != 0):
                lmbda = a[i, k] / a[k, k]
                a[i, k:n] = a[i, k:n] - lmbda * a[k, k:n]
                b[i] = b[i] - lmbda * b[k]
            if verbose:
                print(a, b)

def gauss_elimination_pivot(a, b, verbose=False):
    n, m = shape(a)
    n2 = shape(b)[0]
    assert (n == n2)
    s = zeros(n)
    for i in range(n):
        s[i] = max(abs(a[i, :]))
    for k in range(n - 1):
        # New in pivot version
        p = argmax(abs(a[k:, k]) / s[k:]) + k
        swap(a, p, k)
        swap(b, p, k)
        swap(s, p, k)
        for i in range(k + 1, n):
            assert (a[k, k] != 0)
            if (a[i, k] != 0):
                lmbda = a[i, k] / a[k, k]
                a[i, k:n] = a[i, k:n] - lmbda * a[k, k:n]
                b[i] = b[i] - lmbda * b[k]
            if verbose:
                print(a, b)

def gauss_substitution(a, b):
    n, m = shape(a)
    n2 = shape(b)[0]
    assert (n == n2)
    x = zeros(n)
    for i in range(n - 1, -1, -1):  # decreasing index
        x[i] = (b[i] - dot(a[i, i + 1:], x[i + 1:])) / a[i, i]
    return x

def gauss_pivot(a, b):
    gauss_elimination_pivot(a, b)
    return gauss_substitution(a, b)

def gauss(a, b):
    gauss_elimination(a, b)
    return gauss_substitution(a, b)

def fit_poly_2(points):
    '''
       This function finds a polynomial P of degree 2 that passes
       through the 3 points contained in list 'points'. It returns a numpy
       array containing the coefficient of a polynomial P: array([a0, a1, a2]),
       where P(x) = a0 + a1*x + a2*x**2. Every (x, y) in 'points' must
       verify y = a0 + a1*x + a2*x**2.
    '''

    p1, p2, p3 = points

    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    assert (abs(x1 * y2 - x2 * y1) > 1e-12)

    a = array([
        [1, p1[0], p1[0] ** 2],
        [1, p2[0], p2[0] ** 2],
        [1, p3[0], p3[0] ** 2]
    ])
    b = array([
        [p1[1]],
        [p2[1]],
        [p3[1]]
    ])
    return gauss_pivot(a, b)

def fit_poly(points):
    '''
      This function is a generalization of the previous one.
      Parameters: points is a Python list of pairs representing 2D points.
    '''

    points_set = set(points)

    if (len(points_set) != len(points)):
        raise AssertionError

    a = []
    b = []
    element = []
    for i in points:
        for n in range(len(points)):
            element.append(i[0] ** n)
        a.append(element.copy())
        b.append([i[1]])
        element.clear()

    a = array(a)
    b = array(b)
    return gauss_pivot(a, b)

def gauss_multiple(a, b):
    '''
      This function returns the solution of the system written as
            AX=B, where A is an n x n square matrix, and X and B are n x m matrices.
    '''

    solution = []
    current_b = []
    for i in b.T:
        current_b = []
        for j in i:
            current_b.append(j)
        solution.append(gauss_pivot(a.copy(), current_b))
    solution = transpose(solution)
    return solution

def gauss_multiple_pivot(a, b):
    '''
      This function returns the same result as the previous one,
            except that it uses scaled row pivoting.
    '''

    solution = []
    current_b = []
    for i in b.T:
        current_b = []
        for j in i:
            current_b.append(j)
        solution.append(gauss_pivot(a.copy(), current_b))
    solution = transpose(solution)
    return solution

def matrix_invert(m):
    '''
      This function returns the inverse of the square matrix a passed
            as a paramter.
    '''

    n = shape(m)[0]
    b = identity(n)
    return gauss_multiple_pivot(m, b)

def jacobian(f, x):
    '''
    Returns the Jacobian matrix of f taken in x J(x)
    '''
    n = len(x)
    jac = zeros((n, n))
    h = 10E-4
    fx = f(x)
    # go through the columns of J
    for j in range(n):
        # compute x + h ej
        old_xj = x[j]
        x[j] =x[j]+ h
        # update the Jacobian matrix (eq 3)
        # Now x is x + h*ej
        jac[:, j] = (f(x)-fx) / h
        # restore x[j]
        x[j] = old_xj
    return jac

def newton_raphson_system(f, init_x, epsilon=10E-4, max_iterations=100):
    '''
    Return a solution of f(x)=0 by Newton-Raphson method.
    init_x is the initial guess of the solution
    '''
    x = init_x
    for i in range(max_iterations):
        J = jacobian(f, x)
        delta_x = gauss_pivot(J, -f(x)) # we could also use our functions from Chapter 2!
        x = x + delta_x
        if sqrt(sum(delta_x**2)) <= epsilon:
            return x
    raise Exception("Could not find root!")


#test data
x_data = array([-2.2, -0.3, 0.8, 1.9])
y_data = array([15.180, 10.962, 1.920, -2.040])

def polynomial_fit(x_data, y_data, m):
    '''
    Returns the ai
    '''
    # x_power[i] will contain sum_i x_i^k, k = 0, 2m
    m += 1
    x_powers = zeros(2*m)
    b = zeros(m)
    for i in range(2*m):
        x_powers[i] = sum(x_data**i)
        if i < m:
            b[i] = sum(y_data*x_data**i)
    a = zeros((m, m))
    for k in range(m):
        for j in range(m):
            a[k, j] = x_powers[j+k]
    return gauss_pivot(a, b)

def eval_p(a, x):
    '''
    Returns P(x) where the coefficients of P are in a
    '''
    n = len(a)
    p = a[n-1]
    for i in range(2, n+1):
        p = a[n-i] + x*p
    return p


def eval_p_dp_ddp(a, x):
    '''
    Returns P(x), P'(x) and P''(x) where the coefficients of P are in a
    '''
    n = len(a)
    p = a[n-1]
    dp = 0
    ddp = 0
    for i in range(2, n+1):
        # careful with the order!
        ddp = 2*dp + x*ddp
        dp = p + x*dp
        p = a[n-i] + x*p
    return p, dp, ddp



def trapezoid(f, a, b, n):
    '''
    Integrates f between a and b using n panels (n+1 points)
    '''
    h = (b - a) / n
    x = a + h * arange(n + 1)
    I = f(x[0]) / 2
    for i in range(1, n):
        I += f(x[i])
    I += f(x[n]) / 2
    return h * I


def sin(x):
    return sin(x) / x


def newton_coeffs(x_data, y_data):
    '''
    Returns the coefficients of the Newton polynomial
    '''
    a = y_data.copy()
    m = x_data.size
    assert (m == y_data.size)
    for k in range(1, m):  # go through columns of the table
        for i in range(k, m):  # go through the lines below the diagonal
            a[i] = (a[i] - a[k - 1]) / (x_data[i] - x_data[k - 1])
    return a


def eval_poly_newton(a, x_data, x):
    n = len(x_data) - 1
    p = a[-1]  # last element in a
    for i in range(1, n + 1):
        p = a[n - i] + (x - x_data[n - i]) * p
    return p


def runge_kutta_4(F, x0, y0, x, h):
    '''
    Returns y(x) given the following initial value problem:
    y' = F(x, y)
    y(x0) = y0 # initial conditions
    h is the increment of x used in integration
    F = [y'[0], y'[1], ..., y'[n-1]]
    y = [y[0], y[1], ..., y[n-1]]
    '''

    X = []
    Y = []
    X.append(x0)
    Y.append(y0)
    while x0 < x:
        k0 = F(x0, y0)
        k1 = F(x0 + h / 2.0, y0 + h / 2.0 * k0)
        k2 = F(x0 + h / 2.0, y0 + h / 2 * k1)
        k3 = F(x0 + h, y0 + h * k2)
        y0 = y0 + h / 6.0 * (k0 + 2 * k1 + 2.0 * k2 + k3)
        x0 += h
        X.append(x0)
        Y.append(y0)
    return array(X), array(Y)


def shooting_o2(F, a, alpha, b, beta, u0, u1, delta=10E-3):

    def r(u):
        '''
        Boundary residual, as in equation (1)
        '''
        # Estimate theta_u
        # Evaluate y and y' until x=b, using initial condition y(a)=alpha and y'(a)=u
        X, Y = runge_kutta_4(F, a, array([alpha, u]), b, 0.2)
        theta_u = Y[-1, 0]  # last row, first column (y)
        return theta_u - beta

    # Find u as a the zero of r
    u, _ = false_position(r, u0, u1, delta)

    # Now use u to solve the initial value problem one more time
    X, Y = runge_kutta_4(F, a, array([alpha, u]), b, 0.01)
    return X, Y


def false_position(f, a, b, delta_x):
    '''
    f is the function for which we will find a zero
    a and b define the bracket
    delta_x is the desired accuracy
    Returns ci such that |ci-c_{i-1}| < delta_x
    '''
    fa = f(a)
    fb = f(b)
    if sign(fa) == sign(fb):
        raise Exception("Root hasn't been bracketed")
    estimates = []
    while True:
        c = (a * fb - b * fa) / (fb - fa)
        estimates.append(c)
        fc = f(c)
        if sign(fc) == sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc
        if len(estimates) >= 2 and abs(estimates[-1] - estimates[-2]) <= delta_x:
            break
    return c, estimates


def F(x, y):
    return array([y[1], -1 / x * y[1]])




