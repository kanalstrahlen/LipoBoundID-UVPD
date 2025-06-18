"""
Analysers for LipoBoundID
"""

import pandas as pd
from itertools import product
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class MS2Analyser:
    def __init__(self):
        self.masses = {
            'H': 1.007825,
            'C': 12.0,
            'O': 15.994915,
            'P': 30.973762,
            'N': 14.003074,
            'Na': 22.989221,
        }

        self.headgroups = {
            'PE': {'C': 2, 'H': 5, 'O': 2, 'N': 1, 'P': 1},  # Phosphatidylethanolamine
            'PG': {'C': 3, 'H': 6, 'O': 4, 'P': 1}           # Phosphatidylglycerol
        }

        self.pos_theoretical_masses = self.generate_theoretical_mz(charge='pos')
        self.neg_theoretical_masses = self.generate_theoretical_mz(charge='neg')


    def analyse_spectrum(self, spectrum_peaks, polarity):
        if polarity == 'pos':
            matched_peaks = self.match_peaks(
                spectrum_peaks,
                self.pos_theoretical_masses,
                ppm_threshold=10,
                mass_correction=0,
                polarity='Positive'
            )
        elif polarity == 'neg':
            matched_peaks = self.match_peaks(
                spectrum_peaks,
                self.neg_theoretical_masses,
                ppm_threshold=10,
                mass_correction=21, # to do remove
                polarity='Negative'
            )

        return matched_peaks


    def calculate_mass(self, composition):
        """Calculate exact mass from atomic composition"""
        return sum(count * self.masses[element] for element, count in composition.items())

    
    def generate_theoretical_mz(self, charge='both', min_length=12, max_length=24, max_db=2):
        """
        Generate theoretical m/z values for different lipid classes
        
        Args:
            charge: Ionization mode ('pos', 'neg', or 'both')
            min_length: Minimum fatty acid carbon length
            max_length: Maximum fatty acid carbon length
            max_db: Maximum number of double bonds
            
        Returns:
            DataFrame of theoretical m/z values
        """
        theoretical_masses = []
        
        # generate all possible fatty acid combinations
        chain_lengths = range(2 * min_length, 2 * max_length + 1)
        double_bonds = range(max_db + 1)
        combinations = list(product(chain_lengths, double_bonds))
        
        # calculate masses for each lipid class and chain combination
        for lipid_class, headgroup in self.headgroups.items():
            for length, db in combinations:
                # calculate fatty acid composition (2 chains)
                fa_c = length  # carbon atoms in fatty acids
                fa_h = (length * 2) - db * 2  # hydrogen atoms (subtract 2H per double bond)
                fa_o = 4  # oxygen atoms (2 per fatty acid)
                
                # add glycerol backbone (C3H5O2)
                total_composition = {
                    'C': fa_c + 3 + headgroup.get('C', 0),
                    'H': fa_h + 5 + headgroup.get('H', 0),
                    'O': fa_o + 2 + headgroup.get('O', 0),
                    'P': headgroup.get('P', 0),
                    'N': headgroup.get('N', 0)
                }

                neutral_mass = self.calculate_mass(total_composition)

                modes = []
                if charge == 'pos' or charge == 'both':
                    modes.append('pos')
                if charge == 'neg' or charge == 'both':
                    modes.append('neg')
                
                for mode in modes:
                    if mode == 'neg':
                        # [M-H]- : subtract mass of one proton
                        neg_mass = neutral_mass - self.masses['H']
                        
                        theoretical_masses.append({
                            'lipid_class': lipid_class,
                            'total_length': length,
                            'double_bonds': db,
                            'neutral_mass': neutral_mass,
                            'neg_mz': neg_mass,
                            'pos_mz': None,
                            'sodiated_mz': None,
                            'composition': total_composition.copy()
                        })
                    
                    elif mode == 'pos':
                        # [M+H]+ : add mass of one proton
                        pos_mass = neutral_mass + self.masses['H']
                        # [M+Na]+ : add mass of sodium
                        sodiated_mass = neutral_mass + self.masses['Na']
                        
                        theoretical_masses.append({
                            'lipid_class': lipid_class,
                            'total_length': length,
                            'double_bonds': db,
                            'neutral_mass': neutral_mass,
                            'neg_mz': None,
                            'pos_mz': pos_mass,
                            'sodiated_mz': sodiated_mass,
                            'composition': total_composition.copy()
                        })
        
        return pd.DataFrame(theoretical_masses)

    
    def match_peaks(self, experimental_peaks, theoretical_peaks, ppm_threshold=20, mass_correction=0, polarity='Unknown'):
        """
        Match experimental peaks with theoretical masses using linear calibration
        
        Args:
            experimental_peaks: DataFrame of experimental peaks
            theoretical_peaks: DataFrame of theoretical masses
            ppm_threshold: Maximum mass difference in ppm
            mass_correction: Initial mass correction factor in ppm
            
        Returns:
            DataFrame of matched peaks with linear calibration applied
        """
        matches = []
        
        if mass_correction != 0:
            experimental_peaks['mz'] = experimental_peaks['mz'] * (1 + mass_correction / 1e6)
        
        for _, exp_peak in experimental_peaks.iterrows():
            exp_mz = exp_peak['mz']
            for mz_col in ['pos_mz', 'neg_mz', 'sodiated_mz']:
                valid_mask = theoretical_peaks[mz_col].notna()
                if not valid_mask.any():
                    continue
                    
                valid_theories = theoretical_peaks[valid_mask]
                ppm_diff = (valid_theories[mz_col] - exp_mz) / exp_mz * 1e6
                
                matches_mask = abs(ppm_diff) <= ppm_threshold
                if matches_mask.any():
                    matched_theories = valid_theories[matches_mask].copy()
                    matched_theories['experimental_mz'] = exp_mz
                    matched_theories['intensity'] = exp_peak['intensity']
                    matched_theories['ppm_diff'] = ppm_diff[matches_mask]
                    matched_theories['matched_mode'] = mz_col.replace('_mz', '')
                    matches.append(matched_theories)
        
        if not matches:
            return pd.DataFrame()
        
        combined_df = pd.concat(matches, ignore_index=True)
        
        x = combined_df['experimental_mz'].values
        y = combined_df['ppm_diff'].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        x_cal = np.linspace(min(x), max(x), 20)
        y_cal = slope * x_cal + intercept
        y_upper = y_cal + 5
        y_lower = y_cal - 5

        #plt.figure()
        #plt.gcf().canvas.manager.set_window_title(f'LipoBoundID MS2 Calibration Plot - {polarity} Mode')
        #plt.scatter(x, y, color='black')
        #plt.plot(x_cal, y_cal, color='black')
        #plt.plot(x_cal, y_upper, linestyle='--', color='black')
        #plt.plot(x_cal, y_lower, linestyle='--', color='black')
        #plt.xlabel('m/z')
        #plt.ylabel('Mass error (ppm)')
        #plt.savefig('')


        
        combined_df['predicted_ppm_error'] = slope * combined_df['experimental_mz'] + intercept
        combined_df['calibration_residual'] = combined_df['ppm_diff'] - combined_df['predicted_ppm_error']

        filtered_df = combined_df[abs(combined_df['calibration_residual']) <= 5]  # to do remove hard code
        
        filtered_df.attrs['calibration_slope'] = slope
        filtered_df.attrs['calibration_intercept'] = intercept
        filtered_df.attrs['calibration_r_squared'] = r_value**2
        
        return filtered_df



class HCDAnalyser:
# to do add other lipid classes
    def __init__(self):
        pass
 

    def process_spectrum(self, spectrum, precursor, lipid_class, total_length, double_bonds, ion_type):
        """
        Processes an HCD spectrum to identify diagnostic ions for lipid confirmation.
        
        Args:
            spectrum: Dictionary containing spectral data including peaks_mz and peaks_intensity
            precursor: m/z value of the precursor ion
            lipid_class: Lipid class (e.g., 'PE', 'PC', 'PG')
            total_length: Total number of carbon atoms in the fatty acid chains
            double_bonds: Total number of double bonds in the fatty acid chains
            ion_type: Ion type ('pos', 'neg', or 'sod' for protonated, deprotonated, or sodiated)
            
        Returns:
            For positive and sodiated modes: Boolean indicating headgroup confirmation
            For negative mode: List of possible fatty acid pair compositions
        """
        peaks_mz = spectrum['peaks_mz']
        peaks_intensity = spectrum['peaks_intensity']

        if ion_type == 'pos':
            return self._process_positive_mode(peaks_mz, peaks_intensity, precursor, lipid_class, total_length, double_bonds)
        elif ion_type == 'neg':
            return self._process_negative_mode(peaks_mz, peaks_intensity, total_length, double_bonds)
        elif ion_type == 'sod':
            return self._process_sodiated_mode(peaks_mz, peaks_intensity, precursor, lipid_class, total_length, double_bonds)

    
    def _process_positive_mode(self, peaks_mz, peaks_intensity, precursor, lipid_class, total_length, double_bonds):
        """
        Processes positive mode HCD spectra to confirm lipid headgroup.
        
        Args:
            peaks_mz: Array of m/z values for the spectrum peaks
            peaks_intensity: Array of intensity values for the spectrum peaks
            precursor: m/z value of the precursor ion
            lipid_class: Lipid class (e.g., 'PE', 'PC', 'PG')
            total_length: Total number of carbon atoms in the fatty acid chains
            double_bonds: Total number of double bonds in the fatty acid chains
            
        Returns:
            Boolean indicating whether the headgroup is confirmed
        """
        if lipid_class == 'PE':
            diag_ion = precursor - 141
            diag_anti_ion = 164
            for mz, intensity in zip(peaks_mz, peaks_intensity):
                if abs(mz - diag_ion) <= 0.2 and intensity > 2:
                    for mz, intensity in zip(peaks_mz, peaks_intensity):
                        if abs(mz - diag_anti_ion) <= 0.2 and intensity > 2:
                            return False
                    return True
            return False

        elif lipid_class == 'PG':
            return False

        elif lipid_class == 'PC':
            diag_ion = 184
            for mz, intensity in zip(peaks_mz, peaks_intensity):
                if abs(mz - diag_ion) <= 0.2 and intensity > 2:
                    return True
            return False
        
        # Default case if lipid class not specifically handled
        return False

    
    def _process_negative_mode(self, peaks_mz, peaks_intensity, total_length, double_bonds):
        """
        Processes negative mode HCD spectra to identify possible fatty acid compositions.
        
        Args:
            peaks_mz: Array of m/z values for the spectrum peaks
            peaks_intensity: Array of intensity values for the spectrum peaks
            total_length: Total number of carbon atoms in the fatty acid chains
            double_bonds: Total number of double bonds in the fatty acid chains
            
        Returns:
            List of tuples containing possible fatty acid pair compositions
        """
        matched_pairs = []
        summed_intens = []
        fatty_acids = self._generate_theoretical_fa_pairs(total_length, double_bonds)
        
        for pair in fatty_acids:
            diag_ion1 = pair[2]
            diag_ion2 = pair[3]
            found_match = False

            for mz1, intensity1 in zip(peaks_mz, peaks_intensity):
                if found_match:
                    break
                if abs(mz1 - diag_ion1) <= 0.2 and intensity1 > 1:
                    for mz2, intensity2 in zip(peaks_mz, peaks_intensity):
                        if abs(mz2 - diag_ion2) <= 0.5 and intensity2 > 1:
                            if intensity1 > 2.5 * intensity2:
                                matched_pairs.append(f'{pair[1]}/{pair[0]}')
                                if pair[1] == pair[0]:
                                    summed_intens.append(round(intensity1))
                                else:
                                    summed_intens.append(round(intensity1+intensity2))
                            elif intensity2 > 2.5 * intensity1:
                                matched_pairs.append(f'{pair[0]}/{pair[1]}')
                                if pair[1] == pair[0]:
                                    summed_intens.append(round(intensity1))
                                else:
                                    summed_intens.append(round(intensity1+intensity2))
                            elif intensity2 == intensity1:
                                matched_pairs.append(f'{pair[0]}/{pair[1]}')
                                if pair[1] == pair[0]:
                                    summed_intens.append(round(intensity1))
                                else:
                                    summed_intens.append(round(intensity1+intensity2))
                            else:
                                matched_pairs.append(f'{pair[0]}_{pair[1]}')
                                if pair[1] == pair[0]:
                                    summed_intens.append(round(intensity1))
                                else:
                                    summed_intens.append(round(intensity1+intensity2))
                            found_match = True
                            break
        summed_intens = [i / max(summed_intens) * 100 for i in summed_intens]
                            
        return matched_pairs, summed_intens

  
    def _process_sodiated_mode(self, peaks_mz, peaks_intensity, precursor, lipid_class, total_length, double_bonds):
        """
        Processes sodiated mode HCD spectra to confirm lipid headgroup.
        
        Args:
            peaks_mz: Array of m/z values for the spectrum peaks
            peaks_intensity: Array of intensity values for the spectrum peaks
            precursor: m/z value of the precursor ion
            lipid_class: Lipid class (e.g., 'PE', 'PG')
            total_length: Total number of carbon atoms in the fatty acid chains
            double_bonds: Total number of double bonds in the fatty acid chains
            
        Returns:
            Boolean indicating whether the headgroup is confirmed
        """
        if lipid_class == 'PE':
            diag_ion1 = precursor - 141
            diag_ion2 = precursor - 43 # 164
            for mz, intensity in zip(peaks_mz, peaks_intensity):
                if abs(mz - diag_ion1) <= 0.2 and intensity > 2:
                    for mz, intensity in zip(peaks_mz, peaks_intensity):
                        if abs(mz - diag_ion2) <= 0.2 and intensity > 2:
                            return True
            return False

        elif lipid_class == 'PG':
            diag_ion1 = precursor - 194
            diag_ion2 = precursor - 172 #195
            for mz, intensity in zip(peaks_mz, peaks_intensity):
                if abs(mz - diag_ion1) <= 0.2 and intensity > 2:
                    for mz, intensity in zip(peaks_mz, peaks_intensity):
                        if abs(mz - diag_ion2) <= 0.2 and intensity > 2:
                            return True
            return False

        return False


    def multiple_spectra_detected(self, lipid_id, precursor, ion_type):
        """
        Reports when multiple HCD spectra are detected for the same precursor.
        
        Args:
            lipid_id: Identifier for the lipid (e.g., 'PC 36:2')
            precursor: m/z value of the precursor ion
            ion_type: Ion type ('protonated', 'deprotonated', or 'sodiated')
        """
        print(f'Multiple HCD spectra detected for the {ion_type} {lipid_id} precursor (m/z {precursor})')
        return


    def no_spectra_detected(self, lipid_id, precursor, ion_type):
        """
        Reports when no HCD spectra are detected for a precursor.
        
        Args:
            lipid_id: Identifier for the lipid (e.g., 'PC 36:2')
            precursor: m/z value of the precursor ion
            ion_type: Ion type ('protonated', 'deprotonated', or 'sodiated')
        """
        #print(f'No HCD spectra detected for the {ion_type} {lipid_id} precursor (m/z {precursor})')
        return


    def _generate_theoretical_fa_pairs(self, total_length, double_bonds):
        """
        Generate all possible pairs of fatty acids given total carbon count and double bond count.
        
        Args:
            total_length: Total number of carbon atoms in the fatty acid pair
            double_bonds: Total number of double bonds in the fatty acid pair
        
        Returns:
            List of tuples, each containing:
                - First fatty acid (carbon:double_bonds)
                - Second fatty acid (carbon:double_bonds)
                - Mass of first acyl ion
                - Mass of second acyl ion
        """
        pairs = []
        
        min_length = 6
        max_length = 30
        
        for carbon1 in range(min_length, min(max_length + 1, total_length - min_length + 1)):
            carbon2 = total_length - carbon1
            if carbon2 < min_length or carbon2 > max_length:
                continue
            if carbon2 < carbon1:
                continue
            for db1 in range(double_bonds + 1):
                db2 = double_bonds - db1
                if db1 > carbon1 // 2 or db2 > carbon2 // 2:
                    continue
                mass1 = self._calculate_acyl_mass(carbon1, db1)
                mass2 = self._calculate_acyl_mass(carbon2, db2)
                fa1 = f"{carbon1}:{db1}"
                fa2 = f"{carbon2}:{db2}"
                pairs.append((fa1, fa2, mass1, mass2))
                if carbon1 == carbon2 and db1 != db2:
                    continue
        return pairs
 

    def _calculate_acyl_mass(self, carbon, double_bonds):
        """
        Calculate the mass of an acyl ion based on carbon count and double bonds.
        
        Args:
            carbon: Number of carbon atoms in the fatty acid
            double_bonds: Number of double bonds in the fatty acid
            
        Returns:
            Theoretical mass of the acyl ion
        """
        mass = carbon * 12.0
        mass += (2 * carbon - 1 - 2 * double_bonds) * 1.008
        mass += 16.0 * 2
        return mass



class UVPDAnalyser:
# to do add other lipid classes and more than 1 double bond per chain
    def __init__(self):
        pass


    def process_spectrum(self, spectrum, precursor, lipid_class, total_length, double_bonds, ion_type):
        peaks_mz = spectrum['peaks_mz']
        peaks_intensity = spectrum['peaks_intensity']

        if ion_type == 'pos':
            return self._process_positive_mode(peaks_mz, peaks_intensity, precursor, total_length, double_bonds)
        elif ion_type == 'neg':
            return self._process_negative_mode(peaks_mz, peaks_intensity, precursor, total_length, double_bonds)


    def _process_positive_mode(self, peaks_mz, peaks_intensity, precursor, total_length, double_bonds):
        """
        Processes positive mode UVPD spectra to identify possible fatty acid compositions and db positions.
        
        Args:
            peaks_mz: Array of m/z values for the spectrum peaks
            peaks_intensity: Array of intensity values for the spectrum peaks
            precursor: Precursor ion m/z value
            total_length: Total number of carbon atoms in the fatty acid chains
            double_bonds: Total number of double bonds in the fatty acid chains
                
        Returns:
            List of tuples containing possible fatty acid pair compositions
        """
        matched_pairs = []
        summed_intens = []
        db_positions = {} 
        fatty_acids = self._generate_theoretical_fa_pairs(total_length, double_bonds)
        
        peak_dict = {}
        for mz, intensity in zip(peaks_mz, peaks_intensity):
            if intensity > 0.002:  # to do check this
                peak_dict[mz] = intensity
        
        for pair in fatty_acids:
            diag_ion1 = precursor - pair[2] - 1.0078
            diag_ion2 = precursor - pair[3] - 1.0078
            diag_ion3 = precursor - pair[2] - 1.0078 + 18.011
            diag_ion4 = precursor - pair[3] - 1.0078 + 18.011
            
            ion1_found = any(abs(mz - diag_ion1) <= 0.2 for mz in peak_dict)
            ion2_found = any(abs(mz - diag_ion2) <= 0.2 for mz in peak_dict)
            ion3_found = any(abs(mz - diag_ion3) <= 0.2 for mz in peak_dict)
            ion4_found = any(abs(mz - diag_ion4) <= 0.2 for mz in peak_dict)
            
            if ion1_found and ion2_found and ion3_found and ion4_found:
                pair_id = f'{pair[0]}_{pair[1]}'
                matched_pairs.append(pair_id)
                db_positions[pair_id] = {'fa1': [], 'fa2': []}

        position_intensities = {}
        
        for pair_id in matched_pairs:
            chain1, chain2 = pair_id.split('_')
            chains = [chain1, chain2]
            position_intensities[pair_id] = {'fa1': {}, 'fa2': {}}
            
            check_fas = []
            for i, chain in enumerate(['fa1', 'fa2']):
                if chains[i] not in check_fas:
                    check_fas.append(chains[i])
                    carbon_count, double_bonds = map(int, chains[i].split(':'))
                    if carbon_count % 2 == 0:        
                        if double_bonds > 0:
                            omegas = range(2, carbon_count-1)
                            
                            ch3_mass = 15.0235
                            ch2_mass = 14.0157
                            h_mass = 1.0078
                            
                            for omega in omegas:
                                db_fragment1 = precursor - ch3_mass - (omega - 2) * ch2_mass - h_mass
                                db_fragment2 = db_fragment1 - 14
                                
                                db_peak1 = None
                                db_peak2 = None
                                
                                for mz in peak_dict:
                                    if abs(mz - db_fragment1) <= 0.2:
                                        db_peak1 = (mz, peak_dict[mz])
                                    if abs(mz - db_fragment2) <= 0.2:
                                        db_peak2 = (mz, peak_dict[mz])
                                
                                db_pair_valid = False
                                combined_intensity = 0
                                
                                if db_peak1 and db_peak2:
                                    intensity_ratio = max(db_peak1[1] / db_peak2[1], db_peak2[1] / db_peak1[1])
                                    if intensity_ratio <= 2.0:
                                        db_pair_valid = True
                                        combined_intensity += db_peak1[1] + db_peak2[1]
                                
                                if db_pair_valid:
                                    n_position = carbon_count - omega
                                    if carbon_count % 2 == 0:
                                        position_label = f'n-{n_position}'
                                    else:
                                        position_label = f'cy-{n_position}'
                                    db_positions[pair_id][chain].append(position_label)
                                    position_intensities[pair_id][chain][position_label] = combined_intensity
                        else:
                            db_positions[pair_id][chain].append('No double bond')
                            position_intensities[pair_id][chain]['No double bond'] = 100  # placeholder value
                    else:
                        db_positions[pair_id][chain].append('Odd chain')
                        position_intensities[pair_id][chain]['Odd chain'] = 100  # placeholder value
    
        for pair_id in position_intensities:
            for chain in ['fa1', 'fa2']:
                if position_intensities[pair_id][chain]:
                    max_intensity = max(position_intensities[pair_id][chain].values())
                    if max_intensity > 0:
                        for position in position_intensities[pair_id][chain]:
                            position_intensities[pair_id][chain][position] = (position_intensities[pair_id][chain][position] / max_intensity) * 100
    
        result_pairs = []
        for pair_id in matched_pairs:
            chain1, chain2 = pair_id.split('_')
            fa1_positions = db_positions[pair_id]['fa1']
            fa2_positions = db_positions[pair_id]['fa2']

            result = {
                'pair': pair_id,
                'fa1': {
                    'chain': chain1,
                    'db_positions': fa1_positions if fa1_positions else ['unknown'],
                    'position_intensities': position_intensities[pair_id]['fa1'] if fa1_positions else {}
                },
                'fa2': {
                    'chain': chain2,
                    'db_positions': fa2_positions if fa2_positions else ['unknown'],
                    'position_intensities': position_intensities[pair_id]['fa2'] if fa2_positions else {}
                }
            }
            result_pairs.append(result)
    
        #print("Matched pairs with double bond positions and normalized intensities:", result_pairs)
        return result_pairs


    def _process_negative_mode(self, peaks_mz, peaks_intensity, precursor, total_length, double_bonds):
        """
        Processes negative mode UVPD spectra to identify possible fatty acid compositions and db positions.
        
        Args:
            peaks_mz: Array of m/z values for the spectrum peaks
            peaks_intensity: Array of intensity values for the spectrum peaks
            precursor: Precursor ion m/z value
            total_length: Total number of carbon atoms in the fatty acid chains
            double_bonds: Total number of double bonds in the fatty acid chains
                
        Returns:
            List of tuples containing possible fatty acid pair compositions
        """
        matched_pairs = []
        summed_intens = []
        db_positions = {} 
        fatty_acids = self._generate_theoretical_fa_pairs(total_length, double_bonds)
        
        peak_dict = {}
        for mz, intensity in zip(peaks_mz, peaks_intensity):
            if intensity > 0.002:
                peak_dict[mz] = intensity
        
        for pair in fatty_acids:
            diag_ion1 = precursor - pair[2] - 1.0078
            diag_ion2 = precursor - pair[3] - 1.0078
            diag_ion3 = precursor - pair[2] - 1.0078 + 18.011
            diag_ion4 = precursor - pair[3] - 1.0078 + 18.011

            ion1_found = any(abs(mz - diag_ion1) <= 0.2 for mz in peak_dict)
            ion2_found = any(abs(mz - diag_ion2) <= 0.2 for mz in peak_dict)
            ion3_found = any(abs(mz - diag_ion3) <= 0.2 for mz in peak_dict)
            ion4_found = any(abs(mz - diag_ion4) <= 0.2 for mz in peak_dict)

            if ion1_found and ion2_found and ion3_found and ion4_found:
                pair_id = f'{pair[0]}_{pair[1]}'
                matched_pairs.append(pair_id)
                db_positions[pair_id] = {'fa1': [], 'fa2': []}

        position_intensities = {}
        
        for pair_id in matched_pairs:

            chain1, chain2 = pair_id.split('_')
            chains = [chain1, chain2]
            position_intensities[pair_id] = {'fa1': {}, 'fa2': {}}
            
            check_fas = []
            for i, chain in enumerate(['fa1', 'fa2']):
                if chains[i] not in check_fas:
                    check_fas.append(chains[i])
                    carbon_count, double_bonds = map(int, chains[i].split(':'))            
                    if double_bonds > 0:
                        omegas = range(2, carbon_count-1)
                        
                        ch3_mass = 15.0235
                        ch2_mass = 14.0157
                        h_mass = 1.0078
                        
                        for omega in omegas:
                            db_fragment1 = precursor - ch3_mass - (omega - 2) * ch2_mass - h_mass
                            db_fragment2 = db_fragment1 - 24
                            
                            cyclo_fragment1 = precursor - ch3_mass - (omega - 3) * ch2_mass - 13.0078
                            cyclo_fragment2 = cyclo_fragment1 - 14.0157
                            
                            db_peak1 = None
                            db_peak2 = None
                            cyclo_peak1 = None
                            cyclo_peak2 = None
                            
                            for mz in peak_dict:
                                if carbon_count % 2 == 0:
                                    if abs(mz - db_fragment1) <= 0.2:
                                        db_peak1 = (mz, peak_dict[mz])
                                    if abs(mz - db_fragment2) <= 0.2:
                                        db_peak2 = (mz, peak_dict[mz])
                                if carbon_count % 2 != 0:
                                    if abs(mz - cyclo_fragment1) <= 0.2:
                                        cyclo_peak1 = (mz, peak_dict[mz])
                                    if abs(mz - cyclo_fragment2) <= 0.2:
                                        cyclo_peak2 = (mz, peak_dict[mz])
    
    
                            db_pair_valid = False
                            cyclo_pair_valid = False
                            combined_intensity = 0
                            
                            if db_peak1 and db_peak2:
                                intensity_ratio = max(db_peak1[1] / db_peak2[1], db_peak2[1] / db_peak1[1])
                                if intensity_ratio <= 5.0:
                                    db_pair_valid = True
                                    combined_intensity += db_peak1[1] + db_peak2[1]
    
    
                            if cyclo_peak1 and cyclo_peak2:
                                intensity_ratio = max(cyclo_peak1[1] / cyclo_peak2[1], cyclo_peak2[1] / cyclo_peak1[1])
                                if intensity_ratio <= 5.0:
                                    cyclo_pair_valid = True
                                    combined_intensity += cyclo_peak1[1] + cyclo_peak2[1]
                            
                            if db_pair_valid or cyclo_pair_valid:
                                n_position = carbon_count - omega
                                if carbon_count % 2 == 0:
                                    position_label = f'n-{n_position}'
                                else:
                                    position_label = f'cy-{n_position}'
                                db_positions[pair_id][chain].append(position_label)
                                position_intensities[pair_id][chain][position_label] = combined_intensity
                    else:
                        db_positions[pair_id][chain].append('No double bond')
                        position_intensities[pair_id][chain]['No double bond'] = 100  # Placeholder value
    
        for pair_id in position_intensities:
            for chain in ['fa1', 'fa2']:
                if position_intensities[pair_id][chain]:
                    max_intensity = max(position_intensities[pair_id][chain].values())
                    if max_intensity > 0:
                        for position in position_intensities[pair_id][chain]:
                            position_intensities[pair_id][chain][position] = (position_intensities[pair_id][chain][position] / max_intensity) * 100
    
        result_pairs = []
        for pair_id in matched_pairs:
            chain1, chain2 = pair_id.split('_')
            fa1_positions = db_positions[pair_id]['fa1']
            fa2_positions = db_positions[pair_id]['fa2']

            result = {
                'pair': pair_id,
                'fa1': {
                    'chain': chain1,
                    'db_positions': fa1_positions if fa1_positions else ['unknown'],
                    'position_intensities': position_intensities[pair_id]['fa1'] if fa1_positions else {}
                },
                'fa2': {
                    'chain': chain2,
                    'db_positions': fa2_positions if fa2_positions else ['unknown'],
                    'position_intensities': position_intensities[pair_id]['fa2'] if fa2_positions else {}
                }
            }
            result_pairs.append(result)
        return result_pairs


    def multiple_spectra_detected(self, lipid_id, precursor, ion_type):
        """
        Reports when multiple UVPD spectra are detected for the same precursor.
        
        Args:
            lipid_id: Identifier for the lipid (e.g., 'PC 36:2')
            precursor: m/z value of the precursor ion
            ion_type: Ion type ('protonated', or 'deprotonated')
        """
        print(f'Multiple UVPD spectra detected for the {ion_type} {lipid_id} precursor (m/z {precursor})')
        return


    def no_spectra_detected(self, lipid_id, precursor, ion_type):
        """
        Reports when no UVPD spectra are detected for a precursor.
        
        Args:
            lipid_id: Identifier for the lipid (e.g., 'PC 36:2')
            precursor: m/z value of the precursor ion
            ion_type: Ion type ('protonated', or 'deprotonated')
        """
        #print(f'No UVPD spectra detected for the {ion_type} {lipid_id} precursor (m/z {precursor})')
        return


    def _calculate_acyl_mass(self, carbon, double_bonds):
        """
        Calculate the mass of an acyl ion based on carbon count and double bonds.
        
        Args:
            carbon: Number of carbon atoms in the fatty acid
            double_bonds: Number of double bonds in the fatty acid
            
        Returns:
            Theoretical mass of the acyl ion
        """
        mass = carbon * 12.0
        mass += (2 * carbon - 1 - 2 * double_bonds) * 1.008
        mass += 16.0 * 2
        return mass


    def _generate_theoretical_fa_pairs(self, total_length, double_bonds):
        """
        Generate all possible non-redundant pairs of fatty acids 
        given total carbon count and double bond count.
    
        Args:
            total_length: Total number of carbon atoms in the fatty acid pair
            double_bonds: Total number of double bonds in the fatty acid pair
    
        Returns:
            List of tuples, each containing:
                - First fatty acid (carbon:double_bonds)
                - Second fatty acid (carbon:double_bonds)
                - Mass of first acyl ion
                - Mass of second acyl ion
        """
        pairs = []
    
        min_length = 6
        max_length = 30
        seen = set()
    
        for carbon1 in range(min_length, min(max_length + 1, total_length - min_length + 1)):
            carbon2 = total_length - carbon1
            if carbon2 < min_length or carbon2 > max_length:
                continue
            for db1 in range(double_bonds + 1):
                db2 = double_bonds - db1
                if db1 > carbon1 // 2 or db2 > carbon2 // 2:
                    continue
    
                # Create canonical ordering
                fa1 = f"{carbon1}:{db1}"
                fa2 = f"{carbon2}:{db2}"
                fa_pair = tuple(sorted([fa1, fa2]))
    
                if fa_pair in seen:
                    continue
                seen.add(fa_pair)
    
                mass1 = self._calculate_acyl_mass(*map(int, fa1.split(":")))
                mass2 = self._calculate_acyl_mass(*map(int, fa2.split(":")))
                pairs.append((fa_pair[0], fa_pair[1], mass1, mass2))
    
        return pairs