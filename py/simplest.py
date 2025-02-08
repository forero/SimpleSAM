import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Halo:
    mass: float  # Solar masses
    redshift: float
    radius: float  # kpc
    progenitors: List['Halo']
    hot_gas_mass: float  # Solar masses
    cold_gas_mass: float  # Solar masses
    stellar_mass: float  # Solar masses
    black_hole_mass: float  # Solar masses
    history: Dict[str, List[float]] = None  # For tracking evolution
    
    def __post_init__(self):
        if self.history is None:
            self.history = {
                'redshift': [self.redshift],
                'stellar_mass': [self.stellar_mass],
                'hot_gas_mass': [self.hot_gas_mass],
                'cold_gas_mass': [self.cold_gas_mass],
                'black_hole_mass': [self.black_hole_mass],
                'total_mass': [self.mass]
            }
    
def print_merger_tree(halo: Halo, level: int = 0):
    """Print the merger tree structure for debugging"""
    indent = "  " * level
    print(f"{indent}Halo: mass={halo.mass:.2e} M☉, z={halo.redshift:.2f}")
    if halo.progenitors:
        print(f"{indent}Number of progenitors: {len(halo.progenitors)}")
        for i, prog in enumerate(halo.progenitors):
            print(f"{indent}Progenitor {i+1}:")
            print_merger_tree(prog, level + 1)

class MergerTreeGenerator:
    def __init__(self, 
                 mass_resolution: float = 1e9,  # Minimum halo mass in solar masses
                 max_redshift: float = 8.0):    # Maximum redshift to track
        self.mass_resolution = mass_resolution
        self.max_redshift = max_redshift
        
    def mass_accretion_rate(self, mass: float, redshift: float) -> float:
        """Calculate mass accretion rate following Neistein & Dekel (2008) model
        Returns: dM/dz in solar masses per unit redshift"""
        # Simple approximation of mass accretion rate
        H0 = 70.0  # km/s/Mpc
        omega_m = 0.3
        
        # Mass accretion rate in M_sun/yr following Neistein & Dekel 2008
        s = -0.59 + 0.077 * redshift  # Slope of the accretion rate
        M12 = mass / 1e12  # Mass in units of 10^12 M_sun
        
        # Convert to dM/dz
        dMdt = 25.0 * M12**(1.1) * (1 + redshift)**(2.5) * np.exp(-0.04*redshift)  # M_sun/yr
        dt_dz = -1.0 / (H0 * np.sqrt(omega_m) * (1 + redshift)**2.5)  # Gyr/z
        
        return dMdt * dt_dz * 1e9  # Convert to M_sun/z
        
    def generate_merger_tree(self, final_mass: float, final_redshift: float = 0) -> Halo:
        """Generate a merger tree using the extended Press-Schechter formalism"""
        print(f"\nGenerating tree node: mass={final_mass:.2e}, z={final_redshift:.2f}")
        
        if final_mass < self.mass_resolution or final_redshift > self.max_redshift:
            print("-> Stopping: mass below resolution or redshift too high")
            return None
        
        # Create the halo first so we can track its history
        halo = Halo(
            mass=final_mass,
            redshift=final_redshift,
            radius=160 * (final_mass / 1e12)**(1/3) / (1 + final_redshift),
            progenitors=[],  # Will be populated later
            hot_gas_mass=final_mass * 0.15,  # Cosmic baryon fraction
            cold_gas_mass=final_mass * 0.05,  # Initial cold gas
            stellar_mass=final_mass * 0.001,  # Initial stars
            black_hole_mass=final_mass * 0.001 * 0.001  # Initial black hole
        )
        
        # Initialize history if not done in __post_init__
        if not hasattr(halo, 'history') or halo.history is None:
            halo.history = {
                'redshift': [final_redshift],
                'stellar_mass': [halo.stellar_mass],
                'hot_gas_mass': [halo.hot_gas_mass],
                'cold_gas_mass': [halo.cold_gas_mass],
                'black_hole_mass': [halo.black_hole_mass],
                'total_mass': [final_mass]
            }
        
        # Splitting probability decreases with redshift and small masses
        split_prob = min(0.7 * np.exp(-final_redshift/3) * (final_mass / 1e12) ** 0.2, 0.95)
        print(f"-> Split probability: {split_prob:.3f}")
        
        if np.random.random() < split_prob and final_redshift < self.max_redshift:
            # Major merger case
            mass_ratio = np.random.beta(2, 5)
            mass_ratio = max(0.1, min(mass_ratio, 0.9))
            
            # Calculate progenitor masses
            mass_1 = final_mass * mass_ratio
            mass_2 = final_mass * (1 - mass_ratio)
            
            print(f"-> Splitting into masses: {mass_1:.2e} and {mass_2:.2e}")
            
            # Only split if both resulting masses are above resolution
            if mass_1 > self.mass_resolution and mass_2 > self.mass_resolution:
                dz = 0.05 * (1 + final_redshift/2)
                new_redshift = final_redshift + dz
                
                print(f"-> Generating progenitors at z={new_redshift:.2f}")
                progenitor_1 = self.generate_merger_tree(mass_1, new_redshift)
                progenitor_2 = self.generate_merger_tree(mass_2, new_redshift)
                
                if progenitor_1 is not None:
                    halo.progenitors.append(progenitor_1)
                    # Copy history from most massive progenitor
                    for key in halo.history:
                        halo.history[key].extend(progenitor_1.history[key])
                if progenitor_2 is not None:
                    halo.progenitors.append(progenitor_2)
                
        else:
            # Smooth accretion case
            dz = 0.05 * (1 + final_redshift/2)
            new_redshift = final_redshift + dz
            
            if new_redshift <= self.max_redshift:
                # Calculate mass at earlier time using accretion rate
                dM_dz = self.mass_accretion_rate(final_mass, final_redshift)
                earlier_mass = max(final_mass - dM_dz * dz, self.mass_resolution)
                
                print(f"-> Smooth accretion: mass at z={new_redshift:.2f} is {earlier_mass:.2e}")
                
                if earlier_mass > self.mass_resolution:
                    progenitor = self.generate_merger_tree(earlier_mass, new_redshift)
                    if progenitor is not None:
                        halo.progenitors = [progenitor]
                        # Copy history from progenitor
                        for key in halo.history:
                            halo.history[key].extend(progenitor.history[key])
        
        print(f"-> Created halo with {len(halo.progenitors)} progenitors")
        return halo

def run_simulation(final_mass: float = 1e12, timestep: float = 0.01):
    print("\nStarting simulation...")
    tree_gen = MergerTreeGenerator()
    merger_tree = tree_gen.generate_merger_tree(final_mass)
    
    print("\nFinal merger tree structure:")
    print_merger_tree(merger_tree)
    
    # Sort history by redshift to ensure chronological order
    if merger_tree and merger_tree.history:
        indices = np.argsort(merger_tree.history['redshift'])[::-1]  # Reverse order for high to low redshift
        for key in merger_tree.history:
            merger_tree.history[key] = [merger_tree.history[key][i] for i in indices]
    
    galaxy_model = GalaxyFormation()
    
    # Process tree from high redshift to low redshift
    def process_halo(halo: Halo):
        if halo is None:
            return
            
        # First process progenitors
        for progenitor in halo.progenitors:
            process_halo(progenitor)
        
        # Evolve this galaxy
        n_steps = 10  # Number of evolution steps per merger tree level
        for _ in range(n_steps):
            galaxy_model.evolve_galaxy(halo, timestep)
        
        # Handle mergers if there are progenitors
        if len(halo.progenitors) > 0:
            for progenitor in halo.progenitors[1:]:
                if progenitor:
                    galaxy_model.process_merger(halo, progenitor)
    
    process_halo(merger_tree)
    print("\nMass evolution along main branch:")
    print_main_branch_masses(merger_tree)
    return merger_tree

class GalaxyFormation:
    def __init__(self):
        # Physical parameters with stronger evolution
        self.cooling_rate_norm = 0.5  # Increased cooling rate normalization
        self.star_formation_efficiency = 0.1  # Increased star formation efficiency
        self.supernova_energy_per_mass = 1e51  # erg per solar mass (typical SN energy)
        self.black_hole_efficiency = 0.1  # Fraction of accreted mass converted to energy
        
    def cooling_rate(self, halo: Halo) -> float:
        """Calculate cooling rate in solar masses per year"""
        # Simplified cooling function inspired by White & Frenk (1991)
        T_vir = 3.6e5 * (halo.mass / 1e12)**(2/3) * (1 + halo.redshift)  # Virial temperature in K
        
        if T_vir < 1e4:
            return 0  # Too cold to cool efficiently
        
        # More efficient cooling rate
        rate = self.cooling_rate_norm * halo.hot_gas_mass * (T_vir / 1e6)**(-0.5)
        return min(rate, halo.hot_gas_mass / 0.1)  # Can cool up to 10% of hot gas per step
    
    def star_formation_rate(self, halo: Halo) -> float:
        """Calculate star formation rate in solar masses per year"""
        # Enhanced Schmidt-Kennicutt law
        dynamical_time = 0.1  # Fixed to 100 Myr for simplicity
        return self.star_formation_efficiency * halo.cold_gas_mass / dynamical_time
    
    def supernova_feedback(self, halo: Halo, stellar_mass_formed: float) -> float:
        """Calculate mass of gas ejected by supernovae"""
        energy = stellar_mass_formed * self.supernova_energy_per_mass
        escape_velocity = 200 * (halo.mass / 1e12)**0.5  # km/s
        ejected_mass = 0.1 * stellar_mass_formed  # Simpler mass loading factor
        return min(ejected_mass, halo.hot_gas_mass * 0.5)  # Can't eject more than 50% of hot gas
    
    def black_hole_feedback(self, halo: Halo) -> float:
        """Calculate mass of gas heated by AGN feedback"""
        # Stronger Bondi-Hoyle accretion
        accretion_rate = 0.01 * (halo.black_hole_mass / 1e8) * (halo.hot_gas_mass / 1e10)
        return min(accretion_rate, halo.hot_gas_mass * 0.1)  # Limited to 10% of hot gas
    
    def evolve_galaxy(self, halo: Halo, dt: float):
        """Evolve galaxy forward in time by dt (in Gyr)"""
        # Cooling
        cooled_mass = self.cooling_rate(halo) * dt
        halo.hot_gas_mass -= cooled_mass
        halo.cold_gas_mass += cooled_mass
        
        # Star formation
        formed_stellar_mass = self.star_formation_rate(halo) * dt
        halo.cold_gas_mass -= formed_stellar_mass
        halo.stellar_mass += formed_stellar_mass
        
        # Black hole growth (simple scaling with stellar mass)
        halo.black_hole_mass = 1e-3 * halo.stellar_mass
        
        # Feedback
        ejected_mass_sn = self.supernova_feedback(halo, formed_stellar_mass)
        ejected_mass_agn = self.black_hole_feedback(halo)
        total_ejected = min(ejected_mass_sn + ejected_mass_agn, halo.hot_gas_mass)
        
        halo.hot_gas_mass -= total_ejected
        
        # Record history
        halo.history['redshift'].append(halo.redshift)
        halo.history['stellar_mass'].append(halo.stellar_mass)
        halo.history['hot_gas_mass'].append(halo.hot_gas_mass)
        halo.history['cold_gas_mass'].append(halo.cold_gas_mass)
        halo.history['black_hole_mass'].append(halo.black_hole_mass)
        halo.history['total_mass'].append(halo.mass)
    
    def process_merger(self, primary: Halo, secondary: Halo):
        """Handle galaxy merger physics"""
        # Simple addition of components
        primary.hot_gas_mass += secondary.hot_gas_mass
        primary.cold_gas_mass += secondary.cold_gas_mass
        primary.stellar_mass += secondary.stellar_mass
        primary.black_hole_mass += secondary.black_hole_mass
        
        # Merger-induced starburst
        starburst_efficiency = 0.3 if primary.mass/secondary.mass < 3 else 0.1
        starburst_mass = starburst_efficiency * (primary.cold_gas_mass + secondary.cold_gas_mass)
        primary.cold_gas_mass -= starburst_mass
        primary.stellar_mass += starburst_mass

def print_main_branch_masses(halo: Halo):
    """Print the dark matter halo mass evolution along the main branch"""
    print("\nDark Matter Halo Mass Evolution along Main Branch:")
    print("------------------------------------------------")
    print("Redshift    Mass [M☉]")
    print("--------    ----------")
    
    # Convert scientific notation to a more readable format
    def format_mass(mass):
        exp = int(np.log10(mass))
        coeff = mass / 10**exp
        return f"{coeff:.2f}e{exp}"
    
    for z, m in zip(halo.history['redshift'], halo.history['total_mass']):
        print(f"{z:8.3f}    {format_mass(m)}")

def run_simulation(final_mass: float = 1e12, timestep: float = 0.005):  # Smaller timestep
    tree_gen = MergerTreeGenerator()
    merger_tree = tree_gen.generate_merger_tree(final_mass)
    
    galaxy_model = GalaxyFormation()
    
    # Process tree from high redshift to low redshift
    def process_halo(halo: Halo):
        if halo is None:
            return
            
        # First process progenitors
        for progenitor in halo.progenitors:
            process_halo(progenitor)
        
        # Evolve this galaxy multiple times to ensure smooth evolution
        n_steps = 10  # Number of evolution steps per merger tree level
        for _ in range(n_steps):
            galaxy_model.evolve_galaxy(halo, timestep)
        
        # Handle mergers if there are progenitors
        if len(halo.progenitors) > 0:
            for progenitor in halo.progenitors[1:]:
                galaxy_model.process_merger(halo, progenitor)
    
    process_halo(merger_tree)
    print_main_branch_masses(merger_tree)  # Print the mass evolution
    return merger_tree

def plot_main_branch_evolution(halo: Halo):
    """Plot the evolution of main properties along the main branch."""
    # Convert redshift to lookback time more accurately
    def redshift_to_lookback(z):
        H0 = 70  # km/s/Mpc
        Om = 0.3  # Matter density
        OL = 0.7  # Dark energy density
        # Hubble time in Gyr
        TH = 977.8 / H0
        
        def integrand(z):
            return 1 / ((1 + z) * np.sqrt(Om * (1 + z)**3 + OL))
        
        # Simple integration using trapezoidal rule
        z_array = np.linspace(0, z, 100)
        dz = z_array[1] - z_array[0]
        integral = np.trapz([integrand(zi) for zi in z_array], z_array)
        return TH * integral
    
    # Calculate lookback times
    lookback_time = [redshift_to_lookback(z) for z in halo.history['redshift']]
    
    # Function to safely take log of array with zeros
    def safe_log10(x):
        min_mass = 1e4  # Set minimum mass to 10^4 solar masses
        return np.log10(np.maximum(np.array(x), min_mass))
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Stellar mass evolution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(lookback_time, safe_log10(halo.history['stellar_mass']), 'r-', label='Stellar Mass')
    ax1.set_xlabel('Lookback Time [Gyr]')
    ax1.set_ylabel('log(M$_*$/M$_\odot$)')
    ax1.legend()
    ax1.grid(True)
    ax1.invert_xaxis()  # Higher lookback times (earlier times) on the left
    
    # Gas mass evolution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(lookback_time, safe_log10(halo.history['hot_gas_mass']), 'b-', label='Hot Gas')
    ax2.plot(lookback_time, safe_log10(halo.history['cold_gas_mass']), 'c-', label='Cold Gas')
    ax2.set_xlabel('Lookback Time [Gyr]')
    ax2.set_ylabel('log(M$_{gas}$/M$_\odot$)')
    ax2.legend()
    ax2.grid(True)
    ax2.invert_xaxis()
    
    # Black hole mass evolution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(lookback_time, safe_log10(halo.history['black_hole_mass']), 'k-', label='Black Hole Mass')
    ax3.set_xlabel('Lookback Time [Gyr]')
    ax3.set_ylabel('log(M$_{BH}$/M$_\odot$)')
    ax3.legend()
    ax3.grid(True)
    ax3.invert_xaxis()
    
    # Mass fractions
    ax4 = fig.add_subplot(gs[1, 1])
    total_mass = np.array(halo.history['total_mass'])
    mask = total_mass > 0
    ax4.plot(np.array(lookback_time)[mask], 
             np.array(halo.history['stellar_mass'])[mask]/total_mass[mask], 
             'r-', label='Stellar Fraction')
    ax4.plot(np.array(lookback_time)[mask], 
             np.array(halo.history['hot_gas_mass'])[mask]/total_mass[mask], 
             'b-', label='Hot Gas Fraction')
    ax4.plot(np.array(lookback_time)[mask], 
             np.array(halo.history['cold_gas_mass'])[mask]/total_mass[mask], 
             'c-', label='Cold Gas Fraction')
    ax4.set_xlabel('Lookback Time [Gyr]')
    ax4.set_ylabel('Mass Fraction')
    ax4.legend()
    ax4.grid(True)
    ax4.invert_xaxis()
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    final_halo = run_simulation(final_mass=1e14)
    #fig = plot_main_branch_evolution(final_halo)
    #plt.show()