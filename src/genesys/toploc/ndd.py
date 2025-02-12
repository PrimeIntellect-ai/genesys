MOD_N = 65497
# TODO: We might need to mod y in case they are above. Its unlikely but possible.


def extended_gcd(a, b):
    """
    Returns (gcd, x, y) where gcd is the greatest common divisor of a and b
    and x, y are coefficients where ax + by = gcd
    """
    if a == 0:
        return (b, 0, 1)
    else:
        gcd, x, y = extended_gcd(b % a, a)
        return (gcd, y - (b // a) * x, x)


def mod_inverse(a, m=MOD_N):
    """
    Returns the modular multiplicative inverse of a under modulo m
    """
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError(f"Modular inverse does not exist for {a} (mod {m})")
    else:
        return x % m


def multiply_polynomials_mod(poly1, poly2, mod=MOD_N):
    """
    Multiply two polynomials with coefficients under given modulus.
    Polynomials are represented as lists of coefficients from lowest to highest degree.
    """
    result = [0] * (len(poly1) + len(poly2) - 1)
    for i, coef1 in enumerate(poly1):
        for j, coef2 in enumerate(poly2):
            result[i + j] = (result[i + j] + coef1 * coef2) % mod
    return result


def compute_newton_coefficients(x, y, mod=MOD_N):
    """
    Calculate coefficients for the interpolation polynomial in standard form
    (powers of x) using modular arithmetic.

    Args:
        x (list): x coordinates of points
        y (list): y coordinates of points
        mod (int): modulus for arithmetic operations

    Returns:
        list: Coefficients of polynomial in standard form (ascending powers of x)
    """
    n = len(x)
    # First calculate Newton coefficients
    newton_coef = [[0] * n for _ in range(n)]

    # First column is y values
    for i in range(n):
        newton_coef[i][0] = y[i] % mod

    # Calculate divided differences
    for j in range(1, n):
        for i in range(n - j):
            numerator = (newton_coef[i + 1][j - 1] - newton_coef[i][j - 1]) % mod
            denominator = (x[i + j] - x[i]) % mod
            inv_denominator = mod_inverse(denominator, mod)
            newton_coef[i][j] = (numerator * inv_denominator) % mod

    newton_coeffs = [newton_coef[0][j] for j in range(n)]

    # Convert Newton form to standard form
    result = [newton_coeffs[0]]  # constant term
    current_term = [1]  # start with 1

    for i in range(n - 1):
        # Multiply current_term by (x - x[i])
        next_term = [(-x[i] % mod)] + [1]  # represents (x - x[i])
        current_term = multiply_polynomials_mod(current_term, next_term, mod)

        # Add newton_coeffs[i+1] * current_term to result
        while len(result) < len(current_term):
            result.append(0)
        for j, coef in enumerate(current_term):
            result[j] = (result[j] + (newton_coeffs[i + 1] * coef)) % mod

    return result


def evaluate_polynomial(coeffs, x_eval, mod=MOD_N):
    """
    Evaluate polynomial at x_eval using Horner's method.
    Coefficients are in ascending power order.

    Args:
        coeffs (list): coefficients in standard form (ascending powers of x)
        x_eval (int): point at which to evaluate the polynomial
        mod (int): modulus for arithmetic operations

    Returns:
        int: Value of polynomial at x_eval (mod MOD_N)
    """
    result = coeffs[-1]
    for coef in reversed(coeffs[:-1]):
        result = (result * x_eval + coef) % mod
    return result
