"""
Main window for the LipoBoundID GUI
"""

import wx
import wx.adv
import wx.grid
import threading
import os
import time
from core.processors import RawFileProcessor, MS2Processor, HCDProcessor, UVPDProcessor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure
from scipy.signal import find_peaks


class LipoBoundIDApp(wx.Frame):
    def __init__(self, parent, title):
        """Initialize the main application window"""
        wx.Frame.__init__(self, parent, title=title, size=(1200, 900))

        self.rawFileProcessor = RawFileProcessor()
        self.ms2Processor = MS2Processor()
        self.hcdProcessor = HCDProcessor()
        self.uvpdProcessor = UVPDProcessor()

        self.spectra = None
        self.ms2_matches = None
        self.hcd_matches = None
        self.uvpd_matches = None
        
        self._create_menu()
        
        self.panel = wx.Panel(self)
        
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel.SetSizer(self.main_sizer)
        
        self.CreateStatusBar()
        self.SetStatusText("Please select an input folder.")
        
        self.Centre()
        self.Show()
    

    def _create_menu(self):
        """Create the application menu bar"""
        menubar = wx.MenuBar()

        file_menu = wx.Menu()
        open_item = file_menu.Append(wx.ID_OPEN, "Select folder", "Select the folder containing all .raw files.")
        file_menu.AppendSeparator()

        self.export_item = file_menu.Append(wx.ID_SAVE, "Export results", "Export results to a CSV file")
        self.export_item.Enable(False)
        
        file_menu.AppendSeparator()
        exit_item = file_menu.Append(wx.ID_EXIT, "Exit", "Exit application")

        help_menu = wx.Menu()
        about_item = help_menu.Append(wx.ID_ABOUT, "About", "About LipoBoundID")

        menubar.Append(file_menu, "&File")
        menubar.Append(help_menu, "&Help")
        self.SetMenuBar(menubar)

        self.Bind(wx.EVT_MENU, self.on_open_folder, open_item)
        self.Bind(wx.EVT_MENU, self.on_export_results, self.export_item)
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)
        self.Bind(wx.EVT_MENU, self.on_about, about_item)


    def create_results_panel(self):
        """Create the panel to display results after processing"""
        for child in self.panel.GetChildren():
            child.Destroy()
        
        results_panel = wx.Panel(self.panel)
        results_sizer = wx.BoxSizer(wx.VERTICAL)
        
        top_panel = wx.Panel(results_panel)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.ms2_panel = wx.Panel(top_panel)
        self.ms2_panel.SetBackgroundColour(wx.Colour(240, 240, 240))
        self.ms2_figure = Figure(figsize=(6, 4))
        self.ms2_canvas = FigureCanvas(self.ms2_panel, -1, self.ms2_figure)
        self.ms2_ax = self.ms2_figure.add_subplot(111)
        
        self.ms2_toolbar = NavigationToolbar(self.ms2_canvas)
        self.ms2_toolbar.Realize()
        
        ms2_sizer = wx.BoxSizer(wx.VERTICAL)
        ms2_sizer.Add(self.ms2_canvas, 1, wx.EXPAND)
        ms2_sizer.Add(self.ms2_toolbar, 0, wx.EXPAND)
        self.ms2_panel.SetSizer(ms2_sizer)
        top_sizer.Add(self.ms2_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        ms3_panel = wx.Panel(top_panel)
        ms3_sizer = wx.GridSizer(rows=2, cols=2, hgap=5, vgap=5)
        
        self.create_ms3_panels(ms3_panel, ms3_sizer)
        
        ms3_panel.SetSizer(ms3_sizer)
        top_sizer.Add(ms3_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        top_panel.SetSizer(top_sizer)
        top_panel.SetMinSize(wx.Size(-1, 400))

        results_sizer.Add(top_panel, 2, wx.EXPAND)
        
        grid_panel = wx.Panel(results_panel)
        grid_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.results_grid = wx.grid.Grid(grid_panel)
        self.results_grid.CreateGrid(0, 0)
        self.results_grid.Bind(wx.grid.EVT_GRID_SELECT_CELL, self.on_grid_select)
        
        grid_sizer.Add(self.results_grid, 1, wx.EXPAND | wx.ALL, 5)
        grid_panel.SetSizer(grid_sizer)
        grid_panel.SetMinSize(wx.Size(-1, 100))

        results_sizer.Add(grid_panel, 1, wx.EXPAND)

        results_panel.SetSizer(results_sizer)

        self.main_sizer.Add(results_panel, 1, wx.EXPAND)

        self.panel.Layout()


    def create_ms3_panels(self, parent, sizer):
        """Create the four MS3 spectrum panels"""
        self.ms3_panels = {}
        self.ms3_figures = {}
        self.ms3_canvases = {}
        self.ms3_axes = {}
        self.ms3_toolbars = {}
        
        titles = [
            "HCD Pos. Mode", "UVPD Pos. Mode", 
            "HCD Neg. Mode", "UVPD Neg. Mode"
        ]
        
        for i, title in enumerate(titles):
            panel = wx.Panel(parent)
            panel.SetBackgroundColour(wx.Colour(240,240,240))
            
            figure = Figure(figsize=(3, 2))
            canvas = FigureCanvas(panel, -1, figure)
            ax = figure.add_subplot(111)
            ax.set_title(title)
            
            # Add toolbar for interactive features
            toolbar = NavigationToolbar(canvas)
            toolbar.Realize()
            
            panel_sizer = wx.BoxSizer(wx.VERTICAL)
            panel_sizer.Add(canvas, 1, wx.EXPAND)
            panel_sizer.Add(toolbar, 0, wx.EXPAND)
            panel.SetSizer(panel_sizer)
            
            sizer.Add(panel, 1, wx.EXPAND)

            self.ms3_panels[title] = panel
            self.ms3_figures[title] = figure
            self.ms3_canvases[title] = canvas
            self.ms3_axes[title] = ax
            self.ms3_toolbars[title] = toolbar


    def on_open_folder(self, event):
        """Open and process all RAW files in a folder"""
        with wx.DirDialog(self, "Select folder with RAW files",
                           style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dir_dialog:
            
            if dir_dialog.ShowModal() == wx.ID_CANCEL:
                return
                
            folder_path = dir_dialog.GetPath()
            
            raw_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                        if f.lower().endswith('.raw')]
            
            if not raw_files:
                wx.MessageBox("No RAW files found in the selected folder.", 
                             "Information", wx.OK | wx.ICON_INFORMATION)
                return
            
            steps = [
                "Loading files...",
                "Analysing MS2 spectra...",
                "Analysing MS3 HCD spectra...",
                "Analysing MS3 UVPD spectra..."
            ]

            dlg = wx.ProgressDialog("Processing RAW files", f"Processing RAW files...",
                                   maximum=len(steps), parent=self)

            def process_thread(raw_files):
                try:
                    all_results = []
                    current_step = 0
                    
                    for step_idx, step in enumerate(steps):
                        current_step = step_idx
                        wx.CallAfter(dlg.Update, current_step, f"{step}")
                        if step == "Loading files...":
                            self.spectra = self.rawFileProcessor.process_batch(raw_files)
                            
                        elif step == "Analysing MS2 spectra...":
                            self.ms2_matches = self.ms2Processor.process_ms2_spectra(self.spectra)
                            
                        elif step == "Analysing MS3 HCD spectra...":
                            self.hcd_matches = self.hcdProcessor.process_hcd_spectra(self.spectra, self.ms2_matches)
                            
                        elif step == "Analysing MS3 UVPD spectra...":
                            self.uvpd_matches = self.uvpdProcessor.process_uvpd_spectra(self.spectra, self.hcd_matches)
                            
                    wx.CallAfter(dlg.Update, len(steps))
                    wx.CallAfter(dlg.Destroy)
                    wx.CallAfter(self.export_item.Enable, True)
                    wx.CallAfter(self.update_results_display)

                except Exception as e:
                    wx.CallAfter(dlg.Destroy)
                    wx.CallAfter(wx.MessageBox, f"Error processing files: {str(e)}", 
                                "Error", wx.OK | wx.ICON_ERROR)

            thread = threading.Thread(target=process_thread, args=(raw_files,))
            thread.daemon = True
            thread.start()
            #process_thread(raw_files)


    def on_export_results(self, event):
        """Export the results dataframe to a CSV file"""
        if self.uvpd_matches is None or self.uvpd_matches.empty:
            wx.MessageBox("No results to export.", "Information", wx.OK | wx.ICON_INFORMATION)
            return
            
        with wx.FileDialog(self, "Save results as CSV",
                            wildcard="CSV files (*.csv)|*.csv",
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
                
            filepath = fileDialog.GetPath()
            try:
                self.uvpd_matches.to_csv(filepath, index=False)
                wx.MessageBox(f"Results exported successfully to {filepath}", 
                             "Export Complete", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.MessageBox(f"Error exporting results: {str(e)}", 
                             "Export Error", wx.OK | wx.ICON_ERROR)


    def update_results_display(self):
        """Update the GUI with results after processing"""
        self.create_results_panel()
        self.display_ms2_spectrum()
        self.populate_results_grid()

        for title, ax in self.ms3_axes.items():
            ax.clear()
            ax.set_title(title, fontsize=10)
            ax.set_xlim((500,1000))
            ax.set_ylim((0,110))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_facecolor('#F0F0F0')
            ax.text(
                0.5, 0.5,
                'No spectrum selected',
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=1)
            )


        for title, ax in self.ms3_axes.items():
            self.ms3_figures[title].set_facecolor('#F0F0F0')
            self.ms3_figures[title].tight_layout()
            self.ms3_canvases[title].draw()

        self.SetStatusText(f"Identified {len(self.uvpd_matches)} potential bound lipids from {len(self.spectra)} spectra.")


    def display_ms2_spectrum(self):
        """Display MS2 spectrum with positive and negative modes"""
        pos_spectrum = None
        neg_spectrum = None

        for spectrum in self.spectra:
            if spectrum['ms_order'] == 2:
                if spectrum['polarity'] == 'pos' and pos_spectrum is None:
                    pos_spectrum = spectrum
                elif spectrum['polarity'] == 'neg' and neg_spectrum is None:
                    neg_spectrum = spectrum
        self.ms2_ax.clear()
        
        max_intensity = 0
        min_mz = float('inf')
        max_mz = 0
        self.highlighted_peaks = []

        if pos_spectrum is not None:
            pos_mz = pos_spectrum['mz']
            pos_intensity = pos_spectrum['intensity']
            
            min_mz = min(min_mz, min(pos_mz))
            max_mz = max(max_mz, max(pos_mz))
            max_intensity = max(max_intensity, max(pos_intensity))
            
            self.ms2_ax.plot(
                pos_mz, pos_intensity,
                color='royalblue',
                label='Positive mode' 
            )
            
            self.pos_ms2_spectrum = pos_spectrum

        if neg_spectrum is not None:
            neg_mz = neg_spectrum['mz']
            neg_intensity = -1 * neg_spectrum['intensity']
            
            min_mz = min(min_mz, min(neg_mz))
            max_mz = max(max_mz, max(neg_mz))
            max_intensity = max(max_intensity, max(-1*neg_intensity))
            
            self.ms2_ax.plot(
                neg_mz, neg_intensity,
                color='crimson',
                label='Negative Mode'
            )

            self.neg_ms2_spectrum = neg_spectrum
        
        padding = 0.05 * (max_intensity)
        self.ms2_ax.set_ylim(-max_intensity - padding, max_intensity + padding)
        self.ms2_ax.set_xlim(min_mz, max_mz)
        
        self.ms2_ax.set_xlabel('m/z')
        self.ms2_ax.set_ylabel('Relative intensity (%)')
        
        yticks = self.ms2_ax.get_yticks()
        self.ms2_ax.set_yticks(yticks)
        self.ms2_ax.set_yticklabels([str(abs(int(tick))) for tick in yticks])

        self.ms2_ax.spines['right'].set_visible(False)
        self.ms2_ax.spines['top'].set_visible(False)
        
        self.ms2_ax.text(0.01, 0.99, "MS2 spectra", 
                         horizontalalignment='left',
                         verticalalignment='top',
                         transform=self.ms2_ax.transAxes)
        
        self.ms2_ax.legend(loc='upper right', facecolor='#F0F0F0')

        self.ms2_figure.set_facecolor('#F0F0F0')
        self.ms2_ax.set_facecolor('#F0F0F0')

        self.ms2_figure.tight_layout()
        self.ms2_canvas.draw()


    def populate_results_grid(self):
        """Populate the results grid with lipid identification data"""
        if self.results_grid.GetNumberRows() > 0:
            self.results_grid.DeleteRows(0, self.results_grid.GetNumberRows())
        if self.results_grid.GetNumberCols() > 0:
            self.results_grid.DeleteCols(0, self.results_grid.GetNumberCols())

        self.results_grid.SetDefaultCellBackgroundColour(wx.Colour(240, 240, 240))
        self.results_grid.EnableEditing(False)

        columns = [
            "ID", "Class", "Acyl", "#DB",
            "+H",  "+Na", "-H",
            "+ve Int.", "-ve Int.",
            "+H HCD Evid.", "+Na HCD Evid.", "-H HCD Evid.", 
            "HCD Acyls", "+H UVPD Support?", "+H UVPD Acyls", 
            "+H UVPD DB", "+H UVPD DB Pos.", "-H UVPD Support?",
            "-H UVPD Acyls", "-H UVPD DB", "-H UVPD DB Pos."
        ]
        
        self.results_grid.AppendCols(len(columns))
        
        for i, col in enumerate(columns):
            self.results_grid.SetColLabelValue(i, col)

        if self.uvpd_matches is not None and not self.uvpd_matches.empty:
            rows = self.uvpd_matches.shape[0]
            self.results_grid.AppendRows(rows)
            
            for row_idx, (_, row_data) in enumerate(self.uvpd_matches.iterrows()):
                self.results_grid.SetCellValue(row_idx, 0, str(row_data['lipid_id']))
                self.results_grid.SetCellValue(row_idx, 1, str(row_data['lipid_class']))
                self.results_grid.SetCellValue(row_idx, 2, str(row_data['total_length']))
                self.results_grid.SetCellValue(row_idx, 3, str(row_data['double_bonds']))
                if row_data['detected_pos']:
                    pos_ppm_error = str(round(row_data['error_pos'], 1)) + " ppm"
                    self.results_grid.SetCellValue(row_idx, 4, pos_ppm_error)
                else:
                    self.results_grid.SetCellValue(row_idx, 4, str(row_data['detected_pos']))
                if row_data['detected_sodiated']:
                    sod_ppm_error = str(round(row_data['error_sodiated'], 1))  + " ppm"
                    self.results_grid.SetCellValue(row_idx, 5, sod_ppm_error)
                else:
                    self.results_grid.SetCellValue(row_idx, 5, str(row_data['detected_sodiated']))
                if row_data['detected_neg']:
                    neg_ppm_error = str(round(row_data['error_neg'], 1))  + " ppm"
                    self.results_grid.SetCellValue(row_idx, 6, neg_ppm_error)
                else:
                    self.results_grid.SetCellValue(row_idx, 6, str(row_data['detected_neg']))
                pos_value = 0 if pd.isna(row_data['intensity_pos']) else row_data['intensity_pos']
                sodiated_value = 0 if pd.isna(row_data['intensity_sodiated']) else row_data['intensity_sodiated']
                sum_value = pos_value + sodiated_value
                self.results_grid.SetCellValue(row_idx, 7, str(round(sum_value, 1)) if sum_value > 0 else "")
                self.results_grid.SetCellValue(row_idx, 8, str(round(row_data['intensity_neg'], 1)) if not pd.isna(row_data['intensity_neg']) else "")

                if row_data['single_pos_hcd_present']:
                    self.results_grid.SetCellValue(row_idx, 9, str(row_data['pos_hcd_headgroup_confirmation']))
                else:
                    self.results_grid.SetCellValue(row_idx, 9, "No/many spectra")
                if row_data['single_sod_hcd_present']:
                    self.results_grid.SetCellValue(row_idx, 10, str(row_data['sod_hcd_headgroup_confirmation']))
                else:
                    self.results_grid.SetCellValue(row_idx, 10, "No/many spectra")
                if row_data['single_neg_hcd_present']:
                    self.results_grid.SetCellValue(row_idx, 11, str(row_data['neg_hcd_headgroup_support']))
                else:
                    self.results_grid.SetCellValue(row_idx, 11, "No/many spectra")
                
                chain_pairs = row_data['neg_hcd_chain_pairs']
                if isinstance(chain_pairs, list) and len(chain_pairs) > 0:
                    self.results_grid.SetCellValue(row_idx, 12, ", ".join(chain_pairs))
                else:
                    self.results_grid.SetCellValue(row_idx, 12, "")

                if row_data['single_pos_uvpd_present']:
                    self.results_grid.SetCellValue(row_idx, 13, str(row_data['pos_uvpd_headgroup_support']))
                else:
                    self.results_grid.SetCellValue(row_idx, 13, "No/many spectra")


                chain_pairs = row_data['pos_uvpd_chain_pairs']
                if isinstance(chain_pairs, list) and len(chain_pairs) > 0:
                    self.results_grid.SetCellValue(row_idx, 14, ", ".join(chain_pairs))
                else:
                    self.results_grid.SetCellValue(row_idx, 14, "")

                if row_data['single_pos_uvpd_present']:
                    if row_data['pos_uvpd_headgroup_support']:    
                        self.results_grid.SetCellValue(row_idx, 15, str(row_data['pos_uvpd_db_localised']))
                    else:
                        pass
                else:
                    pass

                db_positions = row_data['pos_uvpd_db_positions']
                if isinstance(db_positions, list) and len(db_positions) > 0:
                    self.results_grid.SetCellValue(row_idx, 16, ", ".join(db_positions))
                else:
                    self.results_grid.SetCellValue(row_idx, 16, "")


                if row_data['single_neg_uvpd_present']:
                    self.results_grid.SetCellValue(row_idx, 17, str(row_data['neg_uvpd_headgroup_support']))
                else:
                    self.results_grid.SetCellValue(row_idx, 17, "No/many spectra")


                chain_pairs = row_data['neg_uvpd_chain_pairs']
                if isinstance(chain_pairs, list) and len(chain_pairs) > 0:
                    self.results_grid.SetCellValue(row_idx, 18, ", ".join(chain_pairs))
                else:
                    self.results_grid.SetCellValue(row_idx, 18, "")

                if row_data['single_neg_uvpd_present']:
                    if row_data['neg_uvpd_headgroup_support']:    
                        self.results_grid.SetCellValue(row_idx, 19, str(row_data['neg_uvpd_db_localised']))
                    else:
                        pass
                else:
                    pass

                db_positions = row_data['neg_uvpd_db_positions']
                if isinstance(db_positions, list) and len(db_positions) > 0:
                    self.results_grid.SetCellValue(row_idx, 20, ", ".join(db_positions))
                else:
                    self.results_grid.SetCellValue(row_idx, 20, "")

        
        self.results_grid.AutoSizeColumns()
        
        for row in range(self.results_grid.GetNumberRows()):
            for col in range(4,7):
                if self.results_grid.GetCellValue(row, col).lower() == "false":
                    self.results_grid.SetCellBackgroundColour(row, col, wx.Colour(255, 200, 200))
                else:
                    self.results_grid.SetCellBackgroundColour(row, col, wx.Colour(200, 255, 200))
                    
            for col in range(9,12):
                if self.results_grid.GetCellValue(row, col).lower() == "true":
                    self.results_grid.SetCellBackgroundColour(row, col, wx.Colour(200, 255, 200))
                elif self.results_grid.GetCellValue(row, col).lower() == "false":
                    self.results_grid.SetCellBackgroundColour(row, col, wx.Colour(255, 200, 200))

            for col in range(13,14):
                if self.results_grid.GetCellValue(row, col).lower() == "true":
                    self.results_grid.SetCellBackgroundColour(row, col, wx.Colour(200, 255, 200))
                elif self.results_grid.GetCellValue(row, col).lower() == "false":
                    self.results_grid.SetCellBackgroundColour(row, col, wx.Colour(255, 200, 200))

            for col in range(15,16):
                if self.results_grid.GetCellValue(row, col).lower() == "true":
                    self.results_grid.SetCellBackgroundColour(row, col, wx.Colour(200, 255, 200))
                elif self.results_grid.GetCellValue(row, col).lower() == "false":
                    self.results_grid.SetCellBackgroundColour(row, col, wx.Colour(255, 200, 200))

            for col in range(17,18):
                if self.results_grid.GetCellValue(row, col).lower() == "true":
                    self.results_grid.SetCellBackgroundColour(row, col, wx.Colour(200, 255, 200))
                elif self.results_grid.GetCellValue(row, col).lower() == "false":
                    self.results_grid.SetCellBackgroundColour(row, col, wx.Colour(255, 200, 200))

            for col in range(19,20):
                if self.results_grid.GetCellValue(row, col).lower() == "true":
                    self.results_grid.SetCellBackgroundColour(row, col, wx.Colour(200, 255, 200))
                elif self.results_grid.GetCellValue(row, col).lower() == "false":
                    self.results_grid.SetCellBackgroundColour(row, col, wx.Colour(255, 200, 200))


        self.grid_data = SortableGridData(self.results_grid)
        self.grid_data.populate_from_grid()
        
        self.results_grid.Bind(wx.grid.EVT_GRID_LABEL_LEFT_CLICK, self.on_column_click)

    
    def on_column_click(self, event):
        """Handle click on grid column label for sorting"""
        if event.GetRow() == -1:
            col = event.GetCol()
            self.grid_data.sort_by_column(col)
            col_name = self.results_grid.GetColLabelValue(col)
            direction = "ascending" if self.grid_data.sort_ascending else "descending"
            self.SetStatusText(f"Sorted by {col_name} ({direction})")
        
        event.Skip()


    def on_grid_select(self, event):
        """Handle selection of a row in the results grid"""
        row = event.GetRow()
        if row >= 0 and row < self.results_grid.GetNumberRows():
            lipid_id = self.results_grid.GetCellValue(row, 0)
            selected_lipid = self.uvpd_matches[self.uvpd_matches['lipid_id'] == lipid_id]
            if not selected_lipid.empty:
                selected_lipid = selected_lipid.iloc[0]
                self.highlight_ms2_peaks(selected_lipid)
                self.update_ms3_spectra(selected_lipid)
                self.SetStatusText(f"Selected: {lipid_id}")

    
    def highlight_ms2_peaks(self, lipid_data):
        """Highlight MS2 peaks for the selected lipid"""
        for highlight in self.highlighted_peaks:
            highlight.remove()

        for text in self.ms2_ax.texts[1:]:
            text.remove()
            
        self.highlighted_peaks = []
        self.ms2_ax.text(1.0, 0.01, lipid_data['lipid_id'], 
                         horizontalalignment='right',
                         verticalalignment='bottom',
                         transform=self.ms2_ax.transAxes)
        
        if lipid_data['detected_pos'] and not pd.isna(lipid_data['pos_mz']):
            pos_mz = lipid_data['pos_mz']
            intensity = lipid_data['intensity_pos']
            highlight = self.ms2_ax.plot(pos_mz, intensity, 'o', color='black', markersize=3, alpha=0.85)[0]
            self.highlighted_peaks.append(highlight)
            self.ms2_ax.annotate(
                "[M+H]+",
                xy=(pos_mz, intensity),
                xytext=(0, 2), textcoords='offset points',
                ha='right', va='bottom',
                fontsize=10, color='black'
            )

        if lipid_data['detected_neg'] and not pd.isna(lipid_data['neg_mz']):
            neg_mz = lipid_data['neg_mz']
            intensity = -1 * lipid_data['intensity_neg']
            highlight = self.ms2_ax.plot(neg_mz, intensity, 'o', color='black', markersize=3, alpha=0.85)[0]
            self.highlighted_peaks.append(highlight)
            self.ms2_ax.annotate(
                "[M-H]-",
                xy=(neg_mz, intensity),
                xytext=(0, -2), textcoords='offset points',
                ha='left', va='top',
                fontsize=10, color='black'
            )

        if lipid_data['detected_sodiated'] and not pd.isna(lipid_data['sodiated_mz']):
            sod_mz = lipid_data['sodiated_mz']
            intensity = lipid_data['intensity_sodiated']
            highlight = self.ms2_ax.plot(sod_mz, intensity, 'o', color='black', markersize=3, alpha=0.85)[0]
            self.highlighted_peaks.append(highlight)
            self.ms2_ax.annotate(
                "[M+Na]+",
                xy=(sod_mz, intensity),
                xytext=(0, 2), textcoords='offset points',
                ha='left', va='bottom',
                fontsize=10, color='black'
            )

        self.ms2_canvas.draw()

   
    def update_ms3_spectra(self, lipid_data):
        """Update MS3 spectra displays for the selected lipid"""
        for title, ax in self.ms3_axes.items():
            ax.clear()
            ax.set_title(title, fontsize=10)
            ax.set_xlim((500,1000))
            ax.set_ylim((0,110))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)            
        
        if lipid_data['single_pos_hcd_present']:
            self.display_ms3_hcd_pos(lipid_data)
        
        if lipid_data['single_neg_hcd_present']:
            self.display_ms3_hcd_neg(lipid_data)
        
        if lipid_data['single_sod_hcd_present']:
            self.display_ms3_hcd_sod(lipid_data)
        
        if lipid_data['single_pos_uvpd_present']:
            self.display_ms3_uvpd_pos(lipid_data)

        if lipid_data['single_neg_uvpd_present']:
            self.display_ms3_uvpd_neg(lipid_data)

        if not (lipid_data['single_sod_hcd_present'] or lipid_data['single_pos_hcd_present']):
            self.ms3_axes['HCD Pos. Mode'].text(
                0.5, 0.5,
                'No spectrum found',
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.ms3_axes['HCD Pos. Mode'].transAxes,
                bbox=dict(facecolor='white', alpha=1)
            )
        if not lipid_data['single_neg_hcd_present']:
            self.ms3_axes['HCD Neg. Mode'].text(
                0.5, 0.5,
                'No spectrum found',
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.ms3_axes['HCD Neg. Mode'].transAxes,
                bbox=dict(facecolor='white', alpha=1)
            )
        if not lipid_data['single_pos_uvpd_present']:
            self.ms3_axes['UVPD Pos. Mode'].text(
                0.5, 0.5,
                'No spectrum found',
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.ms3_axes['UVPD Pos. Mode'].transAxes,
                bbox=dict(facecolor='white', alpha=1)
            )
        if not lipid_data['single_neg_uvpd_present']:
            self.ms3_axes['UVPD Neg. Mode'].text(
                0.5, 0.5,
                'No spectrum found',
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.ms3_axes['UVPD Neg. Mode'].transAxes,
                bbox=dict(facecolor='white', alpha=1)
            )

        for canvas in self.ms3_canvases.values():
            canvas.draw()
    
    def display_ms3_hcd_pos(self, lipid_data):
        """Display positive mode HCD MS3 spectrum"""
        ax = self.ms3_axes["HCD Pos. Mode"]
        self.ms3_figures["HCD Pos. Mode"].set_facecolor('#F0F0F0')
        ax.set_facecolor('#F0F0F0')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        if hasattr(self, 'spectra') and self.spectra:
            precursor = lipid_data['pos_mz']
            for spectrum in self.spectra:
                if (spectrum['ms_order'] == 3 and 
                    spectrum['ms3_type'] == 'HCD' and 
                    spectrum['polarity'] == 'pos' and
                    abs(spectrum['ms3_precursor'] - precursor) <= 0.7):
                    
                    ax.plot(spectrum['mz'], spectrum['intensity'], color='royalblue', alpha=0.75)
                    
                    if lipid_data['pos_hcd_headgroup_confirmation']:
                        peak_info = self.find_nearest_peak(spectrum, precursor, tolerance=0.05)
                        if peak_info:
                            mz, intensity, _ = peak_info
                            ax.plot(mz, intensity, 
                                  'o', color='black', markersize=3, alpha=0.85)
                            ax.annotate("[M+H]+", 
                                      (mz, intensity),
                                      textcoords="offset points", xytext=(0,5),
                                      ha='center', color='royalblue', fontsize=8)
    
                        lipid_class = lipid_data['lipid_class']
                        if lipid_class == 'PE':
                            diagnostic_ion = precursor - 141.02
                            peak_info = self.find_nearest_peak(spectrum, diagnostic_ion, tolerance=0.05)
                            if peak_info:
                                mz, intensity, _ = peak_info
                                ax.plot(mz, intensity, 
                                      'o', color='black', markersize=3, alpha=0.85)
                                ax.annotate("-141", 
                                          (mz, intensity),
                                          textcoords="offset points", xytext=(0,5),
                                          ha='center', color='royalblue', fontsize=8)
    
                    ax.set_title("HCD Pos. Mode", fontsize=10)
                    ax.set_xlim(min(spectrum['mz']), max(spectrum['mz']))
                    self.ms3_figures["HCD Pos. Mode"].tight_layout()
                    break
    
    
    def display_ms3_hcd_neg(self, lipid_data):
        """Display negative mode HCD MS3 spectrum with fatty acyl anion annotations"""
        ax = self.ms3_axes["HCD Neg. Mode"]
        
        if hasattr(self, 'spectra') and self.spectra:
            precursor = lipid_data['neg_mz']
            for spectrum in self.spectra:
                if (spectrum['ms_order'] == 3 and 
                    spectrum['ms3_type'] == 'HCD' and 
                    spectrum['polarity'] == 'neg' and
                    abs(spectrum['ms3_precursor'] - precursor) <= 0.7):
                    
                    ax.plot(spectrum['mz'], spectrum['intensity'], color='crimson')
    
                    peak_info = self.find_nearest_peak(spectrum, precursor, tolerance=0.05)
                    if peak_info:
                        mz, intensity, _ = peak_info            
                        ax.plot(mz, intensity, 
                              'o', color='black', markersize=3, alpha=0.85)
                        ax.annotate("[M-H]-", 
                                  (mz, intensity),
                                  textcoords="offset points", xytext=(0,5),
                                  ha='center', color='black', fontsize=8)
    
                    chain_pairs = lipid_data['neg_hcd_chain_pairs']
                    intensities = lipid_data['neg_hcd_chain_pair_intensities']
                    
                    if isinstance(chain_pairs, list) and len(chain_pairs) > 0:
                        for i, pair in enumerate(chain_pairs):
                            if i < len(intensities):
                                intensity_text = f" ({round(intensities[i],0)}%)"
                            else:
                                intensity_text = ""
                        
                            fa_chains = self.parse_chain_pair(pair)
                            for chain in fa_chains:
                                fa_mz = self.calculate_fa_anion_mz(chain)
                                if fa_mz:
                                    peak_info = self.find_nearest_peak(spectrum, fa_mz, tolerance=0.05)
                                    if peak_info:
                                        mz, intensity, _ = peak_info
                                        ax.plot(mz, intensity, 'o', color='black', markersize=3, alpha=0.85)
                                        if intensity > 5:
                                            ax.annotate(chain, 
                                                      (mz, intensity),
                                                      textcoords="offset points", xytext=(0,5),
                                                      ha='center', color='black', fontsize=8)

                            ax.text(1.0, 0.9 - i*0.15, f"FA: {pair}{intensity_text}", 
                                   transform=ax.transAxes, fontsize=8,
                                   bbox=dict(facecolor='white', alpha=1.0),
                                   ha='right', va='bottom')
                    
                    ax.set_title("HCD Neg. Mode", fontsize=10)
                    ax.set_xlim(min(spectrum['mz']), max(spectrum['mz']))
                    self.ms3_figures["HCD Neg. Mode"].tight_layout()
                    break
    
    
    def display_ms3_hcd_sod(self, lipid_data):
        """Display sodiated mode HCD MS3 spectrum"""
        ax = self.ms3_axes["HCD Pos. Mode"]
        self.ms3_figures["HCD Pos. Mode"].set_facecolor('#F0F0F0')
        ax.set_facecolor('#F0F0F0')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
        if hasattr(self, 'spectra') and self.spectra:
            precursor = lipid_data['sodiated_mz']
            for spectrum in self.spectra:
                if (spectrum['ms_order'] == 3 and 
                    spectrum['ms3_type'] == 'HCD' and 
                    spectrum['polarity'] == 'pos' and
                    abs(spectrum['ms3_precursor'] - precursor) <= 0.7):
    
                    ax.plot(spectrum['mz'], spectrum['intensity'], color='orange', alpha=0.7)
                    
                    if lipid_data['sod_hcd_headgroup_confirmation']:
                        peak_info = self.find_nearest_peak(spectrum, precursor, tolerance=0.05)
                        if peak_info:
                            mz, intensity, _ = peak_info
                            ax.plot(mz, intensity, 
                                  'o', color='black', markersize=3, alpha=0.85)
                            ax.annotate("[M+Na]+", 
                                      (mz, intensity),
                                      textcoords="offset points", xytext=(0,5),
                                      ha='center', color='orange', fontsize=8)
    
                        lipid_class = lipid_data['lipid_class']
                        if lipid_class == 'PE':
                            diagnostic_ions = [
                                (precursor - 141.02, "-141"),
                                (164.01, "164"),
                                (precursor - 43.04, "-43"),
                                (121, "121"),
                                (precursor - 163.00, "-163")
                            ]
                            for diag_ion, label in diagnostic_ions:
                                peak_info = self.find_nearest_peak(spectrum, diag_ion, tolerance=0.05)
                                if peak_info:
                                    mz, intensity, _ = peak_info
                                    ax.plot(mz, intensity, 
                                          'o', color='black', markersize=3, alpha=0.85)
                                    ax.annotate(label, 
                                              (mz, intensity),
                                              textcoords="offset points", xytext=(0,5),
                                              ha='center', color='orange', fontsize=8)
                        elif lipid_class == 'PG':
                            diagnostic_ions = [
                                (precursor - 172.01, "-172"),
                                (195.00, "195"),
                                (precursor - 194.00, "-194")
                            ]
                            for diag_ion, label in diagnostic_ions:
                                peak_info = self.find_nearest_peak(spectrum, diag_ion, tolerance=0.05)
                                if peak_info:
                                    mz, intensity, _ = peak_info
                                    ax.plot(mz, intensity, 
                                          'o', color='black', markersize=3, alpha=0.85)
                                    ax.annotate(label, 
                                              (mz, intensity),
                                              textcoords="offset points", xytext=(0,5),
                                              ha='center', color='orange', fontsize=8)
    
                    self.ms3_figures["HCD Pos. Mode"].tight_layout()
                    ax.set_xlim(min(spectrum['mz']), max(spectrum['mz']))
                    break


    def display_ms3_uvpd_pos(self, lipid_data):
        """Display positive mode UVPD MS3 spectrum with acyl and double bond annotations"""
        ax = self.ms3_axes["UVPD Pos. Mode"]
        self.ms3_figures["UVPD Pos. Mode"].set_facecolor('#F0F0F0')
        ax.set_facecolor('#F0F0F0')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        if hasattr(self, 'spectra') and self.spectra:
            precursor = lipid_data['pos_mz']
            for spectrum in self.spectra:
                if (spectrum['ms_order'] == 3 and 
                    spectrum['ms3_type'] == 'UVPD' and 
                    spectrum['polarity'] == 'pos' and
                    abs(spectrum['ms3_precursor'] - precursor) <= 0.7):
    
                    peak_indices, _ = find_peaks(spectrum['intensity'])
                    peak_intensities = [spectrum['intensity'][i] for i in peak_indices]
                    sorted_peak_intensities = sorted(peak_intensities, reverse=True)
                    fifth_highest = sorted_peak_intensities[4] if len(sorted_peak_intensities) >= 5 else sorted_peak_intensities[-1]
                    
                    y_max = fifth_highest * 1.5
                    
                    ax.plot(spectrum['mz'], spectrum['intensity'], color='royalblue')
                    
                    ax.set_ylim(0, y_max)
                    
                    # Highlight precursor ion if headgroup is supported
                    if lipid_data['pos_uvpd_headgroup_support']:
                        peak_info = self.find_nearest_peak(spectrum, precursor, tolerance=0.05)
                        if peak_info:
                            mz, intensity = peak_info[0], peak_info[1]
                            ax.plot(mz, y_max * 0.97, 
                                   'o', color='black', markersize=3, alpha=0.85)
                            ax.annotate("[M+H]+", 
                                       (mz, y_max),
                                       textcoords="offset points", xytext=(5,0),
                                       ha='left', va='top', color='black', fontsize=8)
                    
                    # Highlight acyl chain diagnostic ions
                    chain_pairs = lipid_data['pos_uvpd_chain_pairs']
                    if chain_pairs and len(chain_pairs) > 0:
                        for i, pair_str in enumerate(chain_pairs):
                            if '_' in pair_str:
                                chains = pair_str.split('_')
                            elif '/' in pair_str:
                                chains = pair_str.split('/')
                            else:
                                chains = [pair_str]
                                
                            for j, chain in enumerate(chains):
                                # Calculate neutral loss peaks for acyl chains
                                carbon, db = map(int, chain.split(':'))
                                acyl_mass = (carbon * 12.0) + ((2 * carbon - 1 - 2 * db) * 1.008) + (16.0 * 2)
                                
                                # Diagnostic ions for neutral loss of acyl chain
                                diag_ion1 = precursor - acyl_mass - 1.0078  # Direct loss
                                diag_ion2 = diag_ion1 + 18.011  # Loss + water
                                
                                # Find and highlight both diagnostic ions
                                for diag_ion, label in [(diag_ion1, f"-{chain}"), (diag_ion2, f"-{chain}+H₂O")]:
                                    peak_info = self.find_nearest_peak(spectrum, diag_ion, tolerance=0.1)
                                    if peak_info:
                                        mz, intensity = peak_info[0], peak_info[1]
                                        ax.plot(mz, intensity, 
                                               'o', color='black', markersize=3, alpha=0.85)
                                        ax.annotate(label, 
                                                   (mz, intensity),
                                                   textcoords="offset points", xytext=(0,5),
                                                   ha='center', va='bottom', color='black', fontsize=8)
                    
                    if lipid_data['pos_uvpd_db_localised']:
                        db_positions = lipid_data['pos_uvpd_db_positions']
                        for db_pos in db_positions:
                            parts = db_pos.split(':')
                            if len(parts) >= 3:
                                chain = f"{parts[0]}:{parts[1]}"
                                position = parts[2]
                            elif len(parts) == 2:  # format like "16:0_n-7" or "16:0_n9"
                                if '_' in db_pos:
                                    chain, position = db_pos.split('_')
                                else:
                                    chain = parts[0]
                                    position = parts[1]
                            else:
                                continue
                                
                            if position.startswith('n-') or position.startswith('cy-'):
                                try:
                                    n_value = int(position.split('-')[1])
                                    carbon_count = int(chain.split(':')[0])
                                    
                                    omega = carbon_count - n_value
                                    ch3_mass = 15.0235
                                    ch2_mass = 14.0157
                                    h_mass = 1.0078
                                    
                                    db_fragment1 = precursor - ch3_mass - (omega - 2) * ch2_mass - h_mass
                                    db_fragment2 = db_fragment1 - 14
                                    
                                    peak1 = self.find_nearest_peak(spectrum, db_fragment1, tolerance=0.05)
                                    peak2 = self.find_nearest_peak(spectrum, db_fragment2, tolerance=0.05)
                                    
                                    if peak1 and peak2:
                                        mz1, intensity1 = peak1[0], peak1[1] 
                                        mz2, intensity2 = peak2[0], peak2[1]
                                        
                                        ax.plot(mz1, intensity1, 'o', color='black', markersize=3, alpha=0.85)
                                        ax.plot(mz2, intensity2, 'o', color='black', markersize=3, alpha=0.85)

                                        mid_point = (mz1 + mz2) / 2
                                        text = ax.annotate(f'{position}',
                                            xy=(mid_point, max(intensity1, intensity2) * 1.05),
                                            xytext=(0, 0),
                                            textcoords="offset points",
                                            ha='center', va='bottom',
                                            fontsize=8, color='red')
                                        text.set_picker(True)
                                        text.draggable()  
                                except (ValueError, IndexError):
                                    continue
                    
                    ax.set_title("UVPD Pos. Mode", fontsize=10)
                    ax.set_xlim(min(spectrum['mz']), max(spectrum['mz']))
                
                    self.ms3_figures["UVPD Pos. Mode"].tight_layout()
                    break


    def display_ms3_uvpd_neg(self, lipid_data):
        """Display negative mode UVPD MS3 spectrum with acyl and double bond annotations"""
        ax = self.ms3_axes["UVPD Neg. Mode"]
        self.ms3_figures["UVPD Neg. Mode"].set_facecolor('#F0F0F0')
        ax.set_facecolor('#F0F0F0')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        if hasattr(self, 'spectra') and self.spectra:
            precursor = lipid_data['neg_mz']
            for spectrum in self.spectra:
                if (spectrum['ms_order'] == 3 and 
                    spectrum['ms3_type'] == 'UVPD' and 
                    spectrum['polarity'] == 'neg' and
                    abs(spectrum['ms3_precursor'] - precursor) <= 0.7):
    
                    peak_indices, _ = find_peaks(spectrum['intensity'])
                    peak_intensities = [spectrum['intensity'][i] for i in peak_indices]
                    sorted_peak_intensities = sorted(peak_intensities, reverse=True)
                    fifth_highest = sorted_peak_intensities[4] if len(sorted_peak_intensities) >= 5 else sorted_peak_intensities[-1]
                    
                    y_max = fifth_highest * 1.5
                    
                    ax.plot(spectrum['mz'], spectrum['intensity'], color='royalblue')
                    
                    ax.set_ylim(0, y_max)
                    
                    # Highlight precursor ion if headgroup is supported
                    if lipid_data['neg_uvpd_headgroup_support']:
                        peak_info = self.find_nearest_peak(spectrum, precursor, tolerance=0.05)
                        if peak_info:
                            mz, intensity = peak_info[0], peak_info[1]
                            ax.plot(mz, y_max * 0.97, 
                                   'o', color='black', markersize=3, alpha=0.85)
                            ax.annotate("[M+H]+", 
                                       (mz, y_max),
                                       textcoords="offset points", xytext=(5,0),
                                       ha='left', va='top', color='black', fontsize=8)
                    
                    # Highlight acyl chain diagnostic ions
                    chain_pairs = lipid_data['neg_uvpd_chain_pairs']
                    if chain_pairs and len(chain_pairs) > 0:
                        for i, pair_str in enumerate(chain_pairs):
                            if '_' in pair_str:
                                chains = pair_str.split('_')
                            elif '/' in pair_str:
                                chains = pair_str.split('/')
                            else:
                                chains = [pair_str]
                                
                            for j, chain in enumerate(chains):
                                # Calculate neutral loss peaks for acyl chains
                                carbon, db = map(int, chain.split(':'))
                                acyl_mass = (carbon * 12.0) + ((2 * carbon - 1 - 2 * db) * 1.008) + (16.0 * 2)
                                
                                # Diagnostic ions for neutral loss of acyl chain
                                diag_ion1 = precursor - acyl_mass - 1.0078  # Direct loss
                                diag_ion2 = diag_ion1 + 18.011  # Loss + water
                                
                                # Find and highlight both diagnostic ions
                                for diag_ion, label in [(diag_ion1, f"-{chain}"), (diag_ion2, f"-{chain}+H₂O")]:
                                    peak_info = self.find_nearest_peak(spectrum, diag_ion, tolerance=0.1)
                                    if peak_info:
                                        mz, intensity = peak_info[0], peak_info[1]
                                        ax.plot(mz, intensity, 
                                               'o', color='black', markersize=3, alpha=0.85)
                                        ax.annotate(label, 
                                                   (mz, intensity),
                                                   textcoords="offset points", xytext=(0,5),
                                                   ha='center', va='bottom', color='black', fontsize=8)
                    
                    if lipid_data['neg_uvpd_db_localised']:
                        db_positions = lipid_data['neg_uvpd_db_positions']
                        for db_pos in db_positions:
                            parts = db_pos.split(':')
                            if len(parts) >= 3:
                                chain = f"{parts[0]}:{parts[1]}"
                                position = parts[2]
                            elif len(parts) == 2:  # format like "16:0_n-7" or "16:0_n9"
                                if '_' in db_pos:
                                    chain, position = db_pos.split('_')
                                else:
                                    chain = parts[0]
                                    position = parts[1]
                            else:
                                continue
                                
                            if position.startswith('n-') or position.startswith('cy-'):
                                try:
                                    n_value = int(position.split('-')[1])
                                    carbon_count = int(chain.split(':')[0])
                                    
                                    omega = carbon_count - n_value
                                    ch3_mass = 15.0235
                                    ch2_mass = 14.0157
                                    h_mass = 1.0078

                                    if carbon_count % 2 == 0:
                                        db_fragment1 = precursor - ch3_mass - (omega - 2) * ch2_mass - h_mass
                                        db_fragment2 = db_fragment1 - 24                                        
                                        
                                        peak1 = self.find_nearest_peak(spectrum, db_fragment1, tolerance=0.05)
                                        peak2 = self.find_nearest_peak(spectrum, db_fragment2, tolerance=0.05)
                                        
                                        if peak1 and peak2:
                                            mz1, intensity1 = peak1[0], peak1[1] 
                                            mz2, intensity2 = peak2[0], peak2[1]
                                            
                                            ax.plot(mz1, intensity1, 'o', color='black', markersize=3, alpha=0.85)
                                            ax.plot(mz2, intensity2, 'o', color='black', markersize=3, alpha=0.85)
    
                                            mid_point = (mz1 + mz2) / 2
                                            text = ax.annotate(f'{position}',
                                                xy=(mid_point, max(intensity1, intensity2) * 1.05),
                                                xytext=(0, 0),
                                                textcoords="offset points",
                                                ha='center', va='bottom',
                                                fontsize=8, color='red')
                                            text.set_picker(True)
                                            text.draggable()
                                    else:
                                        cyclo_fragment1 = precursor - ch3_mass - (omega - 3) * ch2_mass - 13.0078
                                        cyclo_fragment2 = cyclo_fragment1 - 14.0157

                                        peak1 = self.find_nearest_peak(spectrum, cyclo_fragment1, tolerance=0.05)
                                        peak2 = self.find_nearest_peak(spectrum, cyclo_fragment2, tolerance=0.05)

                                        if peak1 and peak2:
                                            mz1, intensity1 = peak1[0], peak1[1] 
                                            mz2, intensity2 = peak2[0], peak2[1]
                                            
                                            ax.plot(mz1, intensity1, 'o', color='black', markersize=3, alpha=0.85)
                                            ax.plot(mz2, intensity2, 'o', color='black', markersize=3, alpha=0.85)
    
                                            mid_point = (mz1 + mz2) / 2
                                            text = ax.annotate(f'{position}',
                                                xy=(mid_point, max(intensity1, intensity2) * 1.05),
                                                xytext=(0, 0),
                                                textcoords="offset points",
                                                ha='center', va='bottom',
                                                fontsize=8, color='red')
                                            text.set_picker(True)
                                            text.draggable()

                                except (ValueError, IndexError):
                                    continue
                    
                    ax.set_title("UVPD Neg. Mode", fontsize=10)
                    ax.set_xlim(min(spectrum['mz']), max(spectrum['mz']))
                
                    self.ms3_figures["UVPD Neg. Mode"].tight_layout()
                    break


    def find_nearest_peak(self, spectrum, target_mz, tolerance=0.05):
        """
        Find the nearest peak in the spectrum within the specified tolerance.
        Returns (mz, intensity, index) of the peak if found, or None if no peak within tolerance.
        """
        valid_indices = np.where(np.abs(spectrum['mz'] - target_mz) <= tolerance)[0]
        
        if len(valid_indices) == 0:
            return None
        
        intensities = spectrum['intensity'][valid_indices]
        max_intensity_idx = valid_indices[np.argmax(intensities)]
        
        return (spectrum['mz'][max_intensity_idx], 
                spectrum['intensity'][max_intensity_idx], 
                max_intensity_idx)

   
    def parse_chain_pair(self, chain_pair):
        """Parse a chain pair like 'FA 16:0_18:1' or 'FA 16:0/18:1' into individual chains"""
        try:
            if chain_pair.startswith('FA '):
                chain_pair = chain_pair[3:]
            if '_' in chain_pair:
                chains = chain_pair.split('_')
            elif '/' in chain_pair:
                chains = chain_pair.split('/')
            else:
                chains = [chain_pair]
                
            return chains
        except:
            return []

   
    def calculate_fa_anion_mz(self, chain):
        """Calculate m/z for fatty acyl anion based on chain structure (e.g., '16:0')"""
        try:
            parts = chain.split(':')
            if len(parts) != 2:
                return None 
            carbon_count = int(parts[0])
            double_bonds = int(parts[1])
            mz = (12 * carbon_count) + (1.0078 * (2 * carbon_count - 1 - 2 * double_bonds)) + (15.9949 * 2)
            return round(mz, 4)
        except:
            return None


    def on_exit(self, event):
        """Exit the application"""
        self.Close()

    def on_about(self, event):
        """Show about dialog"""
        info = wx.adv.AboutDialogInfo()
        info.SetName("LipoBoundID")
        info.SetVersion("0.0.4")
        info.SetDescription("Deep characterisation of protein-lipid complexes with mass spectrometry")
        info.SetCopyright("(C) 2025")
        info.AddDeveloper("Jack L. Bennett & Carla Kirschbaum")
        
        wx.adv.AboutBox(info)



class SortableGridData:
    """Helper class to handle grid data sorting"""
    def __init__(self, grid):
        self.grid = grid
        self.data = []
        self.sort_column = -1
        self.sort_ascending = True
        
    def populate_from_grid(self):
        """Read data from grid into internal storage"""
        self.data = []
        for row in range(self.grid.GetNumberRows()):
            row_data = []
            for col in range(self.grid.GetNumberCols()):
                value = self.grid.GetCellValue(row, col)
                # Store the original background color as well
                bg_color = self.grid.GetCellBackgroundColour(row, col)
                row_data.append((value, bg_color))
            self.data.append(row_data)
    
    def update_grid(self):
        """Update grid with current (possibly sorted) data"""
        for row_idx, row_data in enumerate(self.data):
            for col_idx, (value, bg_color) in enumerate(row_data):
                self.grid.SetCellValue(row_idx, col_idx, value)
                self.grid.SetCellBackgroundColour(row_idx, col_idx, bg_color)
        self.grid.Refresh()
    
    def sort_by_column(self, col_idx):
        """Sort data by specified column"""
        if col_idx == self.sort_column:
            # If already sorting by this column, reverse the order
            self.sort_ascending = not self.sort_ascending
        else:
            # Otherwise, sort ascending by the new column
            self.sort_column = col_idx
            self.sort_ascending = True
        
        # Sort the data
        self.data.sort(key=lambda row: self._get_sort_key(row[col_idx][0]), 
                       reverse=not self.sort_ascending)
        
        # Update the grid with sorted data
        self.update_grid()
    
    def _get_sort_key(self, value):
        """Convert grid cell value to a sortable key with type safety"""
        # Handle empty values first
        if not value or value.strip() == "":
            return (0, "")  # Sort empty values first
            
        # Handle boolean values
        if value.lower() == 'true':
            return (1, 1)
        elif value.lower() == 'false':
            return (1, 0)
        
        # Try to convert to numeric
        try:
            num_value = float(value)
            return (2, num_value)  # Return as numeric type
        except (ValueError, AttributeError):
            # If not numeric, return as string
            return (3, str(value).lower())  # Sort strings last, case-insensitive