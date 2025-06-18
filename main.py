"""
LipoBoundID UVPD - Main entry point
Deep characterisation of protein-lipid complexes with mass spectrometry 
"""

import wx
import os
from gui.main_window import LipoBoundIDApp


def main():
    """Launch the GUI application"""
    app = wx.App()
    frame = LipoBoundIDApp(None, "LipoBoundID UVPD")
    icon_path = os.path.join(os.path.dirname(__file__), "resources", "icon.ico")
    icon = wx.Icon(icon_path)
    frame.SetIcon(icon)

    app.MainLoop()


if __name__ == "__main__":
    main()