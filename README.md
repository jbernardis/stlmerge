stlmerge
========

merge stl files into amf.xml files via command line

Usage:
  python stlmerge.py file1.stl ... filen.stl output.amf.xml
  
  output file can be loaded into slic3r and slices/printed using multiple colors/materials (make sure output file has .amf.xml suffix or slic3r will reject it)
  
Possible future enhancements:
  raise/lower objects so that z minimum - 0 and nothing is printed mid-air
  offset in x and/or y directioy so things can be centered
