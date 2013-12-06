# This file is part of the Printrun suite.
# 
# Printrun is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Printrun is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Printrun.  If not, see <http://www.gnu.org/licenses/>.

import sys, struct, math, numpy

def cross(v1,v2):
	return [v1[1]*v2[2]-v1[2]*v2[1],v1[2]*v2[0]-v1[0]*v2[2],v1[0]*v2[1]-v1[1]*v2[0]]

def genfacet(v):
	veca=[v[1][0]-v[0][0],v[1][1]-v[0][1],v[1][2]-v[0][2]]
	vecb=[v[2][0]-v[1][0],v[2][1]-v[1][1],v[2][2]-v[1][2]]
	vecx=cross(veca,vecb)
	vlen=math.sqrt(sum(map(lambda x:x*x,vecx)))
	if vlen==0:
		vlen=1
	normal=map(lambda x:x/vlen, vecx)
	return [normal,v]

I=[
	[1,0,0,0],
	[0,1,0,0],
	[0,0,1,0],
	[0,0,0,1]
]

def transpose(matrix):
	return zip(*matrix)
	#return [[v[i] for v in matrix] for i in xrange(len(matrix[0]))]
	
def multmatrix(vector,matrix):
	return map(sum, transpose(map(lambda x:[x[0]*p for p in x[1]], zip(vector, transpose(matrix)))))
	
def applymatrix(facet,matrix=I):
	#return facet
	#return [map(lambda x:-1.0*x,multmatrix(facet[0]+[1],matrix)[:3]),map(lambda x:multmatrix(x+[1],matrix)[:3],facet[1])]
	return genfacet(map(lambda x:multmatrix(x+[1],matrix)[:3],facet[1]))
	
f=[[0,0,0],[[-3.022642, 0.642482, -9.510565],[-3.022642, 0.642482, -9.510565],[-3.022642, 0.642482, -9.510565]]]
m=[
	[1,0,0,0],
	[0,1,0,0],
	[0,0,1,1],
	[0,0,0,1]
]

def emitstl(filename,facets=[],objname="stltool_export",binary=False):
	if filename is None:
		return False
	if binary:
		try:
			f=open(filename,"wb")
		except:
			return False
		
		buf="".join(["\0"]*80)
		buf+=struct.pack("<I",len(facets))
		facetformat=struct.Struct("<ffffffffffffH")
		for i in facets:
			l=list(i[0][:])
			for j in i[1]:
				l+=j[:]
			l+=[0]
			buf+=facetformat.pack(*l)
		f.write(buf)
		f.close()
		return True
		

	try:
		f=open(filename,"w")
	except:
		return False
	
	f.write("solid "+objname+"\n")
	for i in facets:
		f.write("  facet normal "+" ".join(map(str,i[0]))+"\n   outer loop\n")
		for j in i[1]:
			f.write("	vertex "+" ".join(map(str,j))+"\n")
		f.write("   endloop"+"\n")
		f.write("  endfacet"+"\n")
	f.write("endsolid "+objname+"\n")
	f.close()
	return True
		
class stl:
	def __init__(self, filename=None, name=None, zZero=False, xOffset=0, yOffset=0):
		self.facet=[[0,0,0],[[0,0,0],[0,0,0],[0,0,0]]]
		self.facets=[]
		self.facetsminz=[]
		self.facetsmaxz=[]
		
		self.name=name
		self.insolid=0
		self.infacet=0
		self.inloop=0
		self.facetloc=0
		self.translatex = 0
		self.translatey = 0
		self.rotation = 0
		self.scalefactor = 1
		self.id = None
		self.filename = filename
		if filename is not None:
			try:
				self.f=list(open(filename))
			except:
				print "Error opening file %s" % filename
				return
			
			if not self.f[0].startswith("solid"):
				print "Not an ascii stl solid - attempting to parse as binary"
				try:
					f=open(filename,"rb")
				except:
					print "Error opening file %s" % filename
					return
				buf=f.read(84)
				while(len(buf)<84):
					newdata=f.read(84-len(buf))
					if not len(newdata):
						break
					buf+=newdata
				facetcount=struct.unpack_from("<I",buf,80)
				facetformat=struct.Struct("<ffffffffffffH")
				for i in xrange(facetcount[0]):
					buf=f.read(50)
					while(len(buf)<50):
						newdata=f.read(50-len(buf))
						if not len(newdata):
							break
						buf+=newdata
					fd=list(facetformat.unpack(buf))
					self.facet=[fd[:3],[fd[3:6],fd[6:9],fd[9:12]]]
					self.facets+=[self.facet]
					facet=self.facet
					self.facetsminz+=[(min(map(lambda x:x[2], facet[1])),facet)]
					self.facetsmaxz+=[(max(map(lambda x:x[2], facet[1])),facet)]
				f.close()
			else:
				for i in self.f:
					self.parseline(i)
			self.setHull(zZero, xOffset, yOffset)
			
	def setId(self, sid):
		self.id = sid
		
	def getId(self):
		return self.id
	
	def setHull(self, zZero, xOffset, yOffset):
		self.projection = numpy.array([])
		minz = 99999
		for f in self.facets:
			if f[1][0][2] < minz: minz = f[1][0][2]
			if f[1][1][2] < minz: minz = f[1][1][2]
			if f[1][2][2] < minz: minz = f[1][2][2]
			if [f[1][0][0], f[1][0][1]] not in self.projection:
				self.projection=numpy.concatenate((self.projection, [f[1][0][0], f[1][0][1]]))
			if [f[1][1][0], f[1][1][1]] not in self.projection:
				self.projection=numpy.concatenate((self.projection, [f[1][1][0], f[1][1][1]]))
			if [f[1][2][0], f[1][2][1]] not in self.projection:
				self.projection=numpy.concatenate((self.projection, [f[1][2][0], f[1][2][1]]))

		n = len(self.projection)			
		self.projection = self.projection.reshape(n/2,2)
		h = self.qhull(self.projection)
		self.adjustHull(h)
		
		modFacets = False
		if zZero and minz != 0:
			for i in range(len(self.facets)):
				for j in range(3):
					self.facets[i][1][j][2] -= minz
			modFacets = True
			
		if xOffset + yOffset != 0:
			for i in range(len(self.facets)):
				for j in range(3):
					self.facets[i][1][j][0] += xOffset
					self.facets[i][1][j][1] += yOffset
			modFacets = True

		if modFacets:					
			self.facetsminz=[]
			self.facetsmaxz=[]
			for facet in self.facets:
				self.facetsminz+=[(min(map(lambda x:x[2], facet[1])),facet)]
				self.facetsmaxz+=[(max(map(lambda x:x[2], facet[1])),facet)]

	def adjustHull(self, d):
		self.hull = d
		hmin = []
		hmax = []
		
		hmin.append(min([x[0] for x in d]))
		hmin.append(min([x[1] for x in d]))
		hmax.append(max([x[0] for x in d]))
		hmax.append(max([x[1] for x in d]))
		
		self.hxCenter = (hmin[0] + hmax[0])/2.0
		self.hyCenter = (hmin[1] + hmax[1])/2.0
		self.hxSize = hmax[0]-hmin[0]
		self.hySize = hmax[1]-hmin[1]
		self.hArea = self.hxSize * self.hySize
		
	def deltaTranslation(self, dx, dy):
		self.translatex += dx
		self.translatey += dy
		
	def isInside(self, xpoint, ypoint):
		def _det(xvert, yvert):
			xvert = numpy.asfarray(xvert)
			yvert = numpy.asfarray(yvert)
			x_prev = numpy.concatenate(([xvert[-1]], xvert[:-1]))
			y_prev = numpy.concatenate(([yvert[-1]], yvert[:-1]))
			return numpy.sum(yvert * x_prev - xvert * y_prev)
		
		smalld=1e-12

		x = []
		y = []
		for px, py in self.hull:
			y.append(py)
			x.append(px)
			
		n = len(x) - 1
		mindst = None
		for i in range(n):
			x1 = x[i]
			y1 = y[i]
			x21 = x[i + 1] - x1
			y21 = y[i + 1] - y1
			x1p = x1 - xpoint
			y1p = y1 - ypoint

			t = -(x1p * x21 + y1p * y21) / (x21 ** 2 + y21 ** 2)
			if t < 0:
				d = x1p ** 2 + y1p ** 2
				if mindst is None or d < mindst:
					snear = False
					mindst = d
					j = i
			elif t <= 1:
				dx = x1p + t * x21
				dy = y1p + t * y21
				d = dx ** 2 + dy ** 2
				if mindst is None or d < mindst:
					snear = True
					mindst = d
					j = i
		mindst **= 0.5
		if mindst < smalld:
			mindst = 0
		elif snear:
			area = _det([x[j], x[j + 1], xpoint],
							 [y[j], y[j + 1], ypoint])
			mindst = math.copysign(mindst, area)
		else:
			if not j:
				x = x[:-1]
				y = y[:-1]
			area = _det([x[j + 1], x[j], x[j - 1]],
							 [y[j + 1], y[j], y[j - 1]])
			mindst = math.copysign(mindst, area)
		return (mindst<=0)

		
	def deltaRotation(self, da):
		self.rotation += da
		
	def deltaScale(self, sf):
		self.scalefactor *= sf
		
	def applyDeltas(self):
		if self.rotation != 0 or self.scalefactor != 1:
			s1 = self.translate(v=[-self.hxCenter, -self.hyCenter, 0])
			if self.rotation != 0:
				s2 = s1.rotate(v=[0, 0, self.rotation])
				s1 = s2
			if self.scalefactor != 1:
				s2 = s1.scale(v=[self.scalefactor, self.scalefactor, self.scalefactor])
				s1 = s2
				
			s = s1.translate(v=[self.hxCenter, self.hyCenter, 0])
			self.facets = [f for f in s.facets]
			self.insolid = s.insolid
			self.infacet = s.infacet
			self.inloop = s.inloop
			self.facetloc = s.facetloc
			self.facetsminz = [f for f in s.facetsminz]
			self.facetsmaxz = [f for f in s.facetsmaxz]
			
		if self.translatex != 0 or self.translatey != 0:
			s = self.translate(v=[self.translatex, self.translatey, 0])
			self.facets = [f for f in s.facets]
			self.insolid = s.insolid
			self.infacet = s.infacet
			self.inloop = s.inloop
			self.facetloc = s.facetloc
			self.facetsminz = [f for f in s.facetsminz]
			self.facetsmaxz = [f for f in s.facetsmaxz]
	
		self.rotation = 0
		self.translatex = 0
		self.translatey = 0	
		self.scalefactor = 1	
		self.setHull()
		
	def translate(self,v=[0,0,0]):
		matrix=[
		[1,0,0,v[0]],
		[0,1,0,v[1]],
		[0,0,1,v[2]],
		[0,0,0,1]
		]
		return self.transform(matrix)
	
	def rotate(self,v=[0,0,0]):
		z=v[2]
		matrix1=[
		[math.cos(math.radians(z)),-math.sin(math.radians(z)),0,0],
		[math.sin(math.radians(z)),math.cos(math.radians(z)),0,0],
		[0,0,1,0],
		[0,0,0,1]
		]
		y=v[0]
		matrix2=[
		[1,0,0,0],
		[0,math.cos(math.radians(y)),-math.sin(math.radians(y)),0],
		[0,math.sin(math.radians(y)),math.cos(math.radians(y)),0],
		[0,0,0,1]
		]
		x=v[1]
		matrix3=[
		[math.cos(math.radians(x)),0,-math.sin(math.radians(x)),0],
		[0,1,0,0],
		[math.sin(math.radians(x)),0,math.cos(math.radians(x)),0],
		[0,0,0,1]
		]
		return self.transform(matrix1).transform(matrix2).transform(matrix3)
	
	def scale(self,v=[0,0,0]):
		matrix=[
		[v[0],0,0,0],
		[0,v[1],0,0],
		[0,0,v[2],0],
		[0,0,0,1]
		]
		return self.transform(matrix)
		
		
	def transform(self,m=I):
		s=stl()
		s.filename = self.filename
		
		s.facets=[applymatrix(i,m) for i in self.facets]
		s.insolid=0
		s.infacet=0
		s.inloop=0
		s.facetloc=0
		s.name=self.name
		for facet in s.facets:
			s.facetsminz+=[(min(map(lambda x:x[2], facet[1])),facet)]
			s.facetsmaxz+=[(max(map(lambda x:x[2], facet[1])),facet)]
		s.setHull()
		return s
		
		
	def clone(self, name=None):
		s=stl()
		s.filename = self.filename
		
		s.facets=[i for i in self.facets]
		s.insolid=0
		s.infacet=0
		s.inloop=0
		s.facetloc=0
		if name is None:
			s.name=self.name
		else:
			s.name = name
			
		self.facetsminz = [f for f in s.facetsminz]
		self.facetsmaxz = [f for f in s.facetsmaxz]
		s.setHull()
		return s
		
	def export(self,f=sys.stdout):
		f.write("solid "+self.name+"\n")
		for i in self.facets:
			f.write("  facet normal "+" ".join(map(str,i[0]))+"\n")
			f.write("   outer loop"+"\n")
			for j in i[1]:
				f.write("	vertex "+" ".join(map(str,j))+"\n")
			f.write("   endloop"+"\n")
			f.write("  endfacet"+"\n")
		f.write("endsolid "+self.name+"\n")
		f.flush()
		
	def parseline(self,l):
		l=l.strip()
		if l.startswith("solid"):
			self.insolid=1
			
		elif l.startswith("endsolid"):
			self.insolid=0
			return 0
		elif l.startswith("facet normal"):
			l=l.replace(",",".")
			self.infacet=11
			self.facetloc=0
			self.facet=[[0,0,0],[[0,0,0],[0,0,0],[0,0,0]]]
			self.facet[0]=map(float,l.split()[2:])
		elif l.startswith("endfacet"):
			self.infacet=0
			self.facets+=[self.facet]
			facet=self.facet
			self.facetsminz+=[(min(map(lambda x:x[2], facet[1])),facet)]
			self.facetsmaxz+=[(max(map(lambda x:x[2], facet[1])),facet)]
		elif l.startswith("vertex"):
			l=l.replace(",",".")
			self.facet[1][self.facetloc]=map(float,l.split()[1:])
			self.facetloc+=1
		return 1

	def qhull(self, sample):
		link = lambda a,b: numpy.concatenate((a,b[1:]))
		edge = lambda a,b: numpy.concatenate(([a],[b]))
	
		def dome(sample,base): 
			h, t = base
			dists = numpy.dot(sample-h, numpy.dot(((0,-1),(1,0)),(t-h)))
			outer = numpy.repeat(sample, dists>0, axis=0)
			
			if len(outer):
				pivot = sample[numpy.argmax(dists)]
				return link(dome(outer, edge(h, pivot)),
							dome(outer, edge(pivot, t)))
			else:
				return base
	
		if len(sample) > 2:
			axis = sample[:,0]
			base = numpy.take(sample, [numpy.argmin(axis), numpy.argmax(axis)], axis=0)
			return link(dome(sample, base),
						dome(sample, base[::-1]))
		else:
			return sample