#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from General_tools import (
    runExampleButton1,
    runExampleButtonGroup1,
    runExampleButtonGrid1,
    runExampleSlider1,
    runExampleSliderGroup1,
    runExampleSliderPlus1,
    runExampleSliderPlusGroup1,
    runExampleSliderPlusGrid1,
    #runExampleSliders1,
    runExampleButtonMenuOverlay1,
    runExampleSliderAndButtonMenuOverlay1,
)

#runExampleButton1()
#runExampleButtonGroup1()
runExampleButtonGrid1()

#address = runExampleSlider1()
#print(f"reference count = {ctypes.c_long.from_address(address)}")
#runExampleSliderGroup1()
#runExampleSliderPlus1()
#runExampleSliderPlusGroup1()
#runExampleSliderPlusGrid1()
#runExampleButtonMenuOverlay1()
#runExampleSliderAndButtonMenuOverlay1()
