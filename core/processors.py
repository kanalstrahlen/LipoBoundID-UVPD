"""
Processors for LipoBoundID
"""
import os
import re
import pandas as pd
import numpy as np
from pymsfilereader import MSFileReader
from scipy.signal import find_peaks
from core.analysers import MS2Analyser, HCDAnalyser, UVPDAnalyser

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class RawFileProcessor:
    def __init__(self):
        return

   
    def process_batch(self, raw_files):
        """
        Process multiple RAW files and extract their spectra.
        
        Args:
            raw_files: List of paths to RAW files
            
        Returns:
            List of spectra dictionaries
        """
        spectra = []
        for file in raw_files:
            spectrum = self.process_file(file)
            spectra.append(spectrum)
        return spectra


    def process_file(self, file_path, scan_num=1):
        """
        Process a single RAW file and extract its spectrum.
        
        Args:
            file_path: Path to the RAW file
            scan_num: Scan number to process (default: 1)
            
        Returns:
            Dictionary containing spectrum data
        """
        raw_file = MSFileReader(file_path)
        
        spectrum = raw_file.GetMassListFromScanNum(scan_num)
        mz_array, intensity_array = self._normalize_spectrum(spectrum)



        scan_metadata = self._extract_scan_metadata(raw_file, scan_num)

        if scan_metadata['ms3_type'] == 'UVPD':
            peaks_mz, peaks_intensity = self.pick_peaks(mz_array, intensity_array, relative_threshold=0.01, num_peaks=100)
        elif scan_metadata['ms3_type'] == 'HCD':
            peaks_mz, peaks_intensity = self.pick_peaks(mz_array, intensity_array, relative_threshold=0.01, num_peaks=30)            
        else:
            mask1 = mz_array < 850 # to do remove hard code
            mz_array = mz_array[mask1]
            mask2 = mz_array > 650
            mz_array = mz_array[mask2]            
            intensity_array = intensity_array[mask1]
            intensity_array = intensity_array[mask2]
            peaks_mz, peaks_intensity = self.pick_peaks(mz_array, intensity_array, relative_threshold=0.1, num_peaks=3000)
        
        # Combine data and metadata
        result = {
            'mz': mz_array,
            'intensity': intensity_array,
            'peaks_mz': peaks_mz,
            'peaks_intensity': peaks_intensity,
            **scan_metadata
        }
        
        return result

   
    def _normalize_spectrum(self, spectrum):
        """
        Normalize spectrum intensity values.
        
        Args:
            spectrum: Raw spectrum data
            
        Returns:
            Tuple of (mz_array, normalized_intensity_array)
        """
        mz_array = np.array(spectrum[0][0])
        intensity_array = np.array(spectrum[0][1])
        max_intensity = np.max(intensity_array)
        normalized_intensity = intensity_array / max_intensity * 100
        return mz_array, normalized_intensity

    
    def _extract_scan_metadata(self, raw_file, scan_num):
        """
        Extract metadata for a scan.
        
        Args:
            raw_file: MSFileReader object
            scan_num: Scan number
            
        Returns:
            Dictionary of scan metadata
        """
        ms_order = raw_file.GetMSOrderForScanNum(scan_num)
        scan_filter = raw_file.GetFilterForScanNum(scan_num)
        polarity_symbol = scan_filter[5]
        polarity = ('pos' if polarity_symbol == '+' else 'neg')
        
        metadata = {
            'ms_order': ms_order,
            'polarity': polarity,
            'ms3_type': None,
            'ms3_precursor': None
        }
        
        if ms_order == 3:
            metadata['ms3_precursor'] = self.extract_last_float_before_at(scan_filter)
            try:
                metadata['ms3_type'] = raw_file.GetActivationTypeForScanNum(scan_num, MSOrder=3)
            except:
                metadata['ms3_type'] = 'UVPD'  # this is an error in the readout but this hack works
                
        return metadata

    
    def extract_last_float_before_at(self, s):
        """
        Extract the last floating point number before an '@' symbol in a string.
        
        Args:
            s: String to parse
            
        Returns:
            Float value or None if not found
        """
        last_at_index = s.rfind('@')
        if last_at_index == -1:
            return None
        substring_before_at = s[:last_at_index]
        matches = list(re.finditer(r'[-+]?\d*\.\d+|\d+', substring_before_at))
        if matches:
            return float(matches[-1].group())
        return None


    def pick_peaks(self, mz_array, intensity_array, relative_threshold=1, num_peaks=30):
        """
        Perform peak picking on spectrum using relative intensity threshold.
        
        Args:
            mz_array: Array of m/z values
            intensity_array: Array of intensity values (0-100%)
            relative_threshold: Minimum intensity relative to base peak (default: 1%)
            num_peaks: Maximum number of peaks to include (ranked by intensity)
            
        Returns:
            Tuple of (peak_mz_values, peak_intensity_values)
        """        
        peaks, properties = find_peaks(intensity_array, height=relative_threshold)

        peak_intensities = intensity_array[peaks]
        peak_mz_values = mz_array[peaks]
        
        if len(peaks) > num_peaks:
            sorted_indices = np.argsort(peak_intensities)[::-1][:num_peaks]
            peak_mz_values = peak_mz_values[sorted_indices]
            peak_intensities = peak_intensities[sorted_indices]
        return peak_mz_values, peak_intensities



class MS2Processor:
    def __init__(self):
        self.analyser = MS2Analyser()
        return


    def process_ms2_spectra(self, spectra):
        """
        Identifies ms2 spectra and processes them to return a list of lipid compositions present
        
        Args:
            spectra: List of dictionaries each containing spectral data, peaks, and metadata for each spectrum
            
        Returns:
            Dataframe containing a quantitative list of lipids identified in the spectrum
        """
        ms2_spectra = []
        dataframe_list = []

        for spectrum in spectra:
            if spectrum['ms_order'] == 2:
                ms2_spectra.append(spectrum)

        print(f'Found {len(ms2_spectra)} MS2 spectra in the folder.')

        for spectrum in ms2_spectra:
            peaks_mz = spectrum['peaks_mz']
            peaks_intensity = spectrum['peaks_intensity']
            polarity = spectrum['polarity']

            input_spectrum = pd.DataFrame({
                'mz': peaks_mz,
                'intensity': peaks_intensity
                })

            matched_peaks = self.analyser.analyse_spectrum(input_spectrum, polarity)
            dataframe_list.append(matched_peaks)

        combined_matches = pd.concat(dataframe_list, ignore_index=True)

        combined_matches = self._combine_dataframes(combined_matches)

        return combined_matches


    def _combine_dataframes(self, df):
        """
        Combines and summarizes lipid identification results from multiple spectra.
    
        Args:
            df: DataFrame containing all matched lipid peaks
    
        Returns:
            Summarized DataFrame with aggregated lipid information and detection status
        """
        df['lipid_id'] = df['lipid_class'] + ' ' + df['total_length'].astype(str) + ':' + df['double_bonds'].astype(str)
    
        lipid_summary = df.groupby('lipid_id').agg({
            'lipid_class': 'first',
            'total_length': 'first', 
            'double_bonds': 'first',
            'neutral_mass': 'first',
            'pos_mz': lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan,
            'neg_mz': lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan,
            'sodiated_mz': lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan
        }).reset_index()

        for mode in ['pos', 'neg', 'sodiated']:
            lipid_summary[f'detected_{mode}'] = False
            lipid_summary[f'error_{mode}'] = np.nan
            lipid_summary[f'intensity_{mode}'] = np.nan
    
        for idx, row in lipid_summary.iterrows():
            lipid_id = row['lipid_id']
            lipid_instances = df[df['lipid_id'] == lipid_id]
    
            for mode in ['pos', 'neg', 'sodiated']:
                match = lipid_instances[lipid_instances['matched_mode'] == mode]
                if not match.empty:
                    lipid_summary.at[idx, f'detected_{mode}'] = True
                    lipid_summary.at[idx, f'intensity_{mode}'] = match['intensity'].values[0]
                    if 'ppm_diff' in match.columns:
                        lipid_summary.at[idx, f'error_{mode}'] = match['ppm_diff'].values[0]
    
        return lipid_summary


class HCDProcessor:
    def __init__(self):
        self.analyser = HCDAnalyser()

    
    def process_hcd_spectra(self, spectra, ms2_matches):
        """
        Processes HCD spectra to confirm lipid identifications from MS2 analysis.
        
        Args:
            spectra: List of dictionaries, each containing spectral data, peaks, 
                    and metadata for each spectrum
            ms2_matches: DataFrame containing lipid identifications from MS2 analysis
            
        Returns:
            DataFrame with MS2 matches extended with HCD confirmation data including:
            - Presence of HCD spectra for each adduct type (protonated, sodiated, deprotonated)
            - Headgroup confirmation status
            - Chain length options derived from negative mode spectra
        """
        pos_hcd_spectra = []
        neg_hcd_spectra = []

        ms3_matches = ms2_matches.copy()
        ms3_matches['single_pos_hcd_present'] = False
        ms3_matches['single_sod_hcd_present'] = False
        ms3_matches['single_neg_hcd_present'] = False
        ms3_matches['pos_hcd_headgroup_confirmation'] = False
        ms3_matches['sod_hcd_headgroup_confirmation'] = False
        ms3_matches['neg_hcd_headgroup_support'] = False
        ms3_matches['neg_hcd_chain_pairs'] = [[] for _ in range(len(ms3_matches))]
        ms3_matches['neg_hcd_chain_pair_intensities'] = [[] for _ in range(len(ms3_matches))]


        pos_hcd_spectra, neg_hcd_spectra = self._filter_hcd_spectra(spectra)

        for index, row in ms3_matches.iterrows():
            if row['detected_pos']:
                self._process_positive_mode(row, index, pos_hcd_spectra, ms3_matches)
            if row['detected_neg']:
                self._process_negative_mode(row, index, neg_hcd_spectra, ms3_matches)
            if row['detected_sodiated']:
                self._process_sodiated_mode(row, index, pos_hcd_spectra, ms3_matches)

        return ms3_matches


    def _filter_hcd_spectra(self, spectra):
        """
        Filters HCD spectra by polarity from the input spectra list.
        
        Args:
            spectra: List of dictionaries containing spectral data
            
        Returns:
            Tuple of (positive_mode_spectra, negative_mode_spectra)
        """
        pos_hcd_spectra = []
        neg_hcd_spectra = []
        
        for spectrum in spectra:
            if spectrum['ms_order'] == 3 and spectrum['ms3_type'] == 'HCD':
                if spectrum['polarity'] == 'pos':
                    pos_hcd_spectra.append(spectrum)
                elif spectrum['polarity'] == 'neg':
                    neg_hcd_spectra.append(spectrum)
        
        return pos_hcd_spectra, neg_hcd_spectra

    
    def _process_positive_mode(self, row, index, pos_hcd_spectra, ms3_matches):
        """
        Processes positive mode HCD spectra for protonated adducts.
        
        Args:
            row: Current row from the MS2 matches DataFrame
            index: Index of the current row
            pos_hcd_spectra: List of positive mode HCD spectra
            ms3_matches: DataFrame to update with processing results
        """
        check_spectra = []
        precursor = row['pos_mz']
        lipid_id = row['lipid_id']
        lipid_class = row['lipid_class']
        total_length = row['total_length']
        double_bonds = row['double_bonds']

        for spectrum in pos_hcd_spectra:
            if abs(spectrum['ms3_precursor'] - precursor) <= 0.7:
                check_spectra.append(spectrum)

        if len(check_spectra) > 1:
            self.analyser.multiple_spectra_detected(lipid_id, precursor, 'protonated')
        elif len(check_spectra) == 0:
            self.analyser.no_spectra_detected(lipid_id, precursor, 'protonated')
        else:
            ms3_matches.at[index, 'single_pos_hcd_present'] = True
            test_spectrum = check_spectra[0]
            confirm = self.analyser.process_spectrum(
                test_spectrum,
                precursor,
                lipid_class,
                total_length,
                double_bonds,
                'pos'
            )
            if confirm:
                ms3_matches.at[index, 'pos_hcd_headgroup_confirmation'] = True


    def _process_negative_mode(self, row, index, neg_hcd_spectra, ms3_matches):
        """
        Processes negative mode HCD spectra for deprotonated adducts.
        
        Args:
            row: Current row from the MS2 matches DataFrame
            index: Index of the current row
            neg_hcd_spectra: List of negative mode HCD spectra
            ms3_matches: DataFrame to update with processing results
        """
        check_spectra = []
        precursor = row['neg_mz']
        lipid_id = row['lipid_id']
        lipid_class = row['lipid_class']
        total_length = row['total_length']
        double_bonds = row['double_bonds']

        for spectrum in neg_hcd_spectra:
            if abs(spectrum['ms3_precursor'] - precursor) <= 0.7:
                check_spectra.append(spectrum)

        if len(check_spectra) > 1:
            self.analyser.multiple_spectra_detected(lipid_id, precursor, 'deprotonated')
        elif len(check_spectra) == 0:
            self.analyser.no_spectra_detected(lipid_id, precursor, 'deprotonated')
        else:
            ms3_matches.at[index, 'single_neg_hcd_present'] = True
            test_spectrum = check_spectra[0]
            matched_pairs, intensities = self.analyser.process_spectrum(
                test_spectrum,
                precursor,
                lipid_class,
                total_length,
                double_bonds,
                'neg'
            )

            if len(matched_pairs) >= 1:
                ms3_matches.at[index, 'neg_hcd_headgroup_support'] = True
                ms3_matches.at[index, 'neg_hcd_chain_pairs'] = matched_pairs
                ms3_matches.at[index, 'neg_hcd_chain_pair_intensities'] = intensities


    def _process_sodiated_mode(self, row, index, pos_hcd_spectra, ms3_matches):
        """
        Processes positive mode HCD spectra for sodiated adducts.
        
        Args:
            row: Current row from the MS2 matches DataFrame
            index: Index of the current row
            pos_hcd_spectra: List of positive mode HCD spectra
            ms3_matches: DataFrame to update with processing results
        """
        check_spectra = []
        precursor = row['sodiated_mz']
        lipid_id = row['lipid_id']
        lipid_class = row['lipid_class']
        total_length = row['total_length']
        double_bonds = row['double_bonds']

        for spectrum in pos_hcd_spectra:
            if abs(spectrum['ms3_precursor'] - precursor) <= 0.7:
                check_spectra.append(spectrum)

        if len(check_spectra) > 1:
            self.analyser.multiple_spectra_detected(lipid_id, precursor, 'sodiated')
        elif len(check_spectra) == 0:
            self.analyser.no_spectra_detected(lipid_id, precursor, 'sodiated')
        else:
            ms3_matches.at[index, 'single_sod_hcd_present'] = True
            test_spectrum = check_spectra[0]
            confirm = self.analyser.process_spectrum(
                test_spectrum,
                precursor,
                lipid_class,
                total_length,
                double_bonds,
                'sod'
            )
            if confirm:
                ms3_matches.at[index, 'sod_hcd_headgroup_confirmation'] = True



class UVPDProcessor:
    def __init__(self):
        self.analyser = UVPDAnalyser()
        return


    def process_uvpd_spectra(self, spectra, hcd_matches):
        pos_uvpd_spectra = []
        neg_uvpd_spectra = []

        uvpd_matches = hcd_matches.copy()
        uvpd_matches['single_pos_uvpd_present'] = False
        uvpd_matches['pos_uvpd_headgroup_support'] = False
        uvpd_matches['pos_uvpd_chain_pairs'] = [[] for _ in range(len(uvpd_matches))]
        uvpd_matches['pos_uvpd_db_localised'] = False
        uvpd_matches['pos_uvpd_db_positions'] = [[] for _ in range(len(uvpd_matches))]
        uvpd_matches['single_neg_uvpd_present'] = False
        uvpd_matches['neg_uvpd_headgroup_support'] = False
        uvpd_matches['neg_uvpd_chain_pairs'] = [[] for _ in range(len(uvpd_matches))]
        uvpd_matches['neg_uvpd_db_localised'] = False
        uvpd_matches['neg_uvpd_db_positions'] = [[] for _ in range(len(uvpd_matches))]

        pos_uvpd_spectra, neg_uvpd_spectra = self._filter_uvpd_spectra(spectra)

        for index, row in uvpd_matches.iterrows():
            if row['detected_pos']:
                self._process_positive_mode(row, index, pos_uvpd_spectra, uvpd_matches)
            if row['detected_neg']:
                self._process_negative_mode(row, index, neg_uvpd_spectra, uvpd_matches)

        uvpd_matches.to_csv('output3.csv')
        return uvpd_matches


    def _filter_uvpd_spectra(self, spectra):
        """
        Filters UVPD spectra by polarity from the input spectra list.
        
        Args:
            spectra: List of dictionaries containing spectral data
            
        Returns:
            Tuple of (positive_mode_spectra, negative_mode_spectra)
        """
        pos_uvpd_spectra = []
        neg_uvpd_spectra = []
        
        for spectrum in spectra:
            if spectrum['ms_order'] == 3 and spectrum['ms3_type'] == 'UVPD':
                if spectrum['polarity'] == 'pos':
                    pos_uvpd_spectra.append(spectrum)
                elif spectrum['polarity'] == 'neg':
                    neg_uvpd_spectra.append(spectrum)
        
        return pos_uvpd_spectra, neg_uvpd_spectra


    def _process_positive_mode(self, row, index, pos_uvpd_spectra, uvpd_matches):
        """
        Processes positive mode UVPD spectra for protonated adducts.
        
        Args:
            row: Current row from the MS2 matches DataFrame
            index: Index of the current row
            pos_hcd_spectra: List of positive mode UVPD spectra
            uvpd_matches: DataFrame to update with processing results
        """
        check_spectra = []
        precursor = row['pos_mz']
        lipid_id = row['lipid_id']
        lipid_class = row['lipid_class']
        total_length = row['total_length']
        double_bonds = row['double_bonds']

        for spectrum in pos_uvpd_spectra:
            if abs(spectrum['ms3_precursor'] - precursor) <= 0.7:
                check_spectra.append(spectrum)

        if len(check_spectra) > 1:
            self.analyser.multiple_spectra_detected(lipid_id, precursor, 'protonated')
        elif len(check_spectra) == 0:
            self.analyser.no_spectra_detected(lipid_id, precursor, 'protonated')
        else:
            uvpd_matches.at[index, 'single_pos_uvpd_present'] = True
            test_spectrum = check_spectra[0]
            result_pairs = self.analyser.process_spectrum(
                test_spectrum,
                precursor,
                lipid_class,
                total_length,
                double_bonds,
                'pos'
            )

            if result_pairs and len(result_pairs) > 0:
                uvpd_matches.at[index, 'pos_uvpd_headgroup_support'] = True
                uvpd_matches.at[index, 'pos_uvpd_chain_pairs'] = [pair['pair'] for pair in result_pairs]
                
                db_positions = []
                db_intensities = []
                
                for pair in result_pairs:
                    if 'db_positions' in pair['fa1'] and pair['fa1']['db_positions'] != ['unknown']:
                        positions = pair['fa1']['db_positions']
                        if 'position_intensities' in pair['fa1']:
                            intensities = pair['fa1']['position_intensities']
                            for pos in positions:
                                if pos != 'No double bond' and pos in intensities:
                                    db_positions.append(f"{pair['fa1']['chain']}:{pos}")
                                    db_intensities.append(intensities[pos])
                    
                    if 'db_positions' in pair['fa2'] and pair['fa2']['db_positions'] != ['unknown']:
                        positions = pair['fa2']['db_positions']
                        if 'position_intensities' in pair['fa2']:
                            intensities = pair['fa2']['position_intensities']
                            for pos in positions:
                                if pos != 'No double bond' and pos in intensities:
                                    db_positions.append(f"{pair['fa2']['chain']}:{pos}")
                                    db_intensities.append(intensities[pos])
                
                if db_positions:
                    uvpd_matches.at[index, 'pos_uvpd_db_localised'] = True
                    uvpd_matches.at[index, 'pos_uvpd_db_positions'] = db_positions


    def _process_negative_mode(self, row, index, neg_uvpd_spectra, uvpd_matches):
        """
        Processes positive mode UVPD spectra for deprotonated adducts.
        
        Args:
            row: Current row from the MS2 matches DataFrame
            index: Index of the current row
            neg_hcd_spectra: List of positive mode UVPD spectra
            uvpd_matches: DataFrame to update with processing results
        """
        check_spectra = []
        precursor = row['neg_mz']
        lipid_id = row['lipid_id']
        lipid_class = row['lipid_class']
        total_length = row['total_length']
        double_bonds = row['double_bonds']

        for spectrum in neg_uvpd_spectra:
            if abs(spectrum['ms3_precursor'] - precursor) <= 0.7:
                check_spectra.append(spectrum)

        if len(check_spectra) > 1:
            self.analyser.multiple_spectra_detected(lipid_id, precursor, 'deprotonated')
        elif len(check_spectra) == 0:
            self.analyser.no_spectra_detected(lipid_id, precursor, 'deprotonated')
        else:
            uvpd_matches.at[index, 'single_neg_uvpd_present'] = True
            test_spectrum = check_spectra[0]
            result_pairs = self.analyser.process_spectrum(
                test_spectrum,
                precursor,
                lipid_class,
                total_length,
                double_bonds,
                'neg'
            )

            if result_pairs and len(result_pairs) > 0:
                uvpd_matches.at[index, 'neg_uvpd_headgroup_support'] = True
                uvpd_matches.at[index, 'neg_uvpd_chain_pairs'] = [pair['pair'] for pair in result_pairs]
                
                db_positions = []
                db_intensities = []
                
                for pair in result_pairs:
                    if 'db_positions' in pair['fa1'] and pair['fa1']['db_positions'] != ['unknown']:
                        positions = pair['fa1']['db_positions']
                        if 'position_intensities' in pair['fa1']:
                            intensities = pair['fa1']['position_intensities']
                            for pos in positions:
                                if pos != 'No double bond' and pos in intensities:
                                    db_positions.append(f"{pair['fa1']['chain']}:{pos}")
                                    db_intensities.append(intensities[pos])
                    
                    if 'db_positions' in pair['fa2'] and pair['fa2']['db_positions'] != ['unknown']:
                        positions = pair['fa2']['db_positions']
                        if 'position_intensities' in pair['fa2']:
                            intensities = pair['fa2']['position_intensities']
                            for pos in positions:
                                if pos != 'No double bond' and pos in intensities:
                                    db_positions.append(f"{pair['fa2']['chain']}:{pos}")
                                    db_intensities.append(intensities[pos])
                
                if db_positions:
                    uvpd_matches.at[index, 'neg_uvpd_db_localised'] = True
                    uvpd_matches.at[index, 'neg_uvpd_db_positions'] = db_positions