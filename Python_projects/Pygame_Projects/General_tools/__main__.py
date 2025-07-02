#!/usr/bin/env python3
import ctypes
import os
import sys
sys.path.append(os.path.abspath('../../'))

from General_tools import (
    runExampleSlider1,
    runExampleSliderGroup1,
    runExampleSliderPlus1,
    runExampleSliders1,
    runExampleButton1,
    runExampleButtonGrid1,
    runExampleMenuOverlay1,
)

#address = runExampleSlider1()
#print(f"reference count = {ctypes.c_long.from_address(address)}")
runExampleSliderGroup1()
runExampleSliderPlus1()
#runExampleSliders1()
#runExampleButton1()
#runExampleButtonGrid1()
#runExampleMenuOverlay1()
