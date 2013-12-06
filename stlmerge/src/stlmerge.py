'''
Created on Dec 4, 2013

@author: ejefber
'''
import sys
import os
import stltool

class volume:
	def __init__(self, fn, startIdx, zzero, xoffset, yoffset):
		self.vertexMap = {}
		self.vertexVal = []
		self.vertexIdx = startIdx
		self.triangles = []
		self.name = fn
		
		self.stl = stltool.stl(filename=fn, zZero=zzero, xOffset=xoffset, yOffset=yoffset)
		for f in self.stl.facets:
			triangle = [None, None, None]
			for px in range(3):
				key = str(f[1][px][0]) + ";" + str(f[1][px][1]) + ";" + str(f[1][px][2])
				if key not in self.vertexMap.keys():
					self.vertexMap[key] = self.vertexIdx
					self.vertexVal.append(key)
					self.vertexIdx += 1
			
				triangle[px] = self.vertexMap[key]
	
			self.triangles.append(triangle)
			
	def maxVertexIdx(self):
		return self.vertexIdx
	
	def getName(self):
		return self.name
	
	def getVertices(self):
		result = ""
		for v in self.vertexVal:
			x, y, z = v.split(";")
			result += "        <vertex>\n"
			result += "          <coordinates>\n"
			result += "            <x>%s</x>\n" % x
			result += "            <y>%s</y>\n" % y
			result += "            <z>%s</z>\n" % z
			result += "          </coordinates>\n"
			result += "        </vertex>\n"	

		return result
	
	def getTriangles(self):
		result = ""
		for t in self.triangles:
			result += "        <triangle>\n"
			result += "          <v1>%s</v1>\n" % t[0]
			result += "          <v2>%s</v2>\n" % t[1]
			result += "          <v3>%s</v3>\n" % t[2]
			result += "        </triangle>\n"
		
		return result
			
class amf:
	def __init__(self):
		self.volumes = []
		self.vIdx = 0
		
	def addStl(self, fn, zZero=False, xOffset=0, yOffset=0):
		v = volume(fn, self.vIdx, zZero, xOffset, yOffset)
		self.vIdx = v.maxVertexIdx()
		self.volumes.append(v)

	def merge(self):
		result = ""
		
		result += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
		result += "<amf unit=\"millimeter\">\n"
		result += "  <metadata type=\"cad\">stlmerge.py</metadata>\n"
		
		vx = 0
		for v in self.volumes:
			result += "  <material id=\"%d\">\n" % vx
			vx += 1
			result += "    <metadata type=\"Name\">%s</metadata>\n" % v.getName()
			result += "  </material>\n"
		
		result += "  <object id=\"0\">\n"
		result += "    <mesh>\n"
		result += "      <vertices>\n"
		
		for v in self.volumes:
			result += v.getVertices()
			
		result += "      </vertices>\n"
		
		vx = 0
		for v in self.volumes:
			result += "      <volume materialid=\"%d\">\n" % vx
			result += v.getTriangles()
			result += "      </volume>\n"
			vx += 1

		result += "    </mesh>\n"
		result += "  </object>\n"
		result += "</amf>"
		return result

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print "Error: need at least 1 stl file and one output amf xml file"
		print ""
		print "Usage: python stlmerge.py stlfile [stlfile...] amfxmlfile"
		exit(1)
		
	stls = sys.argv[1:]
	del stls[-1]
	
	err = False
	for f in stls:
		if not os.path.isfile(f):
			print "STL file \"%s\" does not exist" % f
			err = True
			
	if err:
		exit(1)
		
	amfFn = sys.argv[-1]
		
	a = amf()
	for s in stls:
		a.addStl(s)

	try:
		f=open(amfFn,"w")
	except:
		print "Unable to open output file %s" % amfFn
		exit(1)
	
	f.write(a.merge())
	f.close()

	print "AMF XML file %s created" % amfFn
	
	exit(0)
	
	
