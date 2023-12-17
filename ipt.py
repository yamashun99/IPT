import numpy as np


def density_of_states(half_bandwidth, energy):
    return 2 / (np.pi * half_bandwidth) * np.sqrt(1 - (energy / half_bandwidth) ** 2)


def bare_green_function(energies, half_bandwidth, small_imaginary_part):
    energy_prime = np.linspace(-half_bandwidth, half_bandwidth, 1000)
    integrand = density_of_states(half_bandwidth, energy_prime).reshape(-1, 1) / (
        energies.reshape(1, -1)
        - energy_prime.reshape(-1, 1)
        + 1j * small_imaginary_part
    )
    return np.trapz(integrand, energy_prime, axis=0)


def green_function(energies, half_bandwidth, small_imaginary_part, S):
    energy_prime = np.linspace(-half_bandwidth, half_bandwidth, 1000)
    integrand = density_of_states(half_bandwidth, energy_prime).reshape(-1, 1) / (
        energies.reshape(1, -1)
        - energy_prime.reshape(-1, 1)
        + 1j * small_imaginary_part
        - S.reshape(1, -1)
    )
    return np.trapz(integrand, energy_prime, axis=0)


def f(x):
    return np.where(x < 0, 1, 0)


def alpha(t, rho, energies):
    integrand = (rho * f(energies)).reshape(-1, 1) * np.exp(
        -1j * energies.reshape(-1, 1) * t.reshape(1, -1)
    )
    return np.trapz(integrand, energies, axis=0)


def beta(t, rho, energies):
    integrand = (rho * f(-energies)).reshape(-1, 1) * np.exp(
        -1j * energies.reshape(-1, 1) * t.reshape(1, -1)
    )
    return np.trapz(integrand, energies, axis=0)


def sigma(U, rho, energies):
    ts = np.linspace(0, 200, 1000).reshape(-1, 1)
    integrand = (
        beta(ts, rho, energies) * alpha(-ts, rho, energies) * beta(ts, rho, energies)
        + alpha(ts, rho, energies) * beta(-ts, rho, energies) * alpha(ts, rho, energies)
    ).reshape(-1, 1) * np.exp(1j * (energies).reshape(1, -1) * ts)
    return -1.0j * U**2 * np.trapz(integrand, ts, axis=0)


def cavity_green_function(G, S):
    return 1 / (1 / G + S)
