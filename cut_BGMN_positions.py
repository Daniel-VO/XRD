"""
Created 18. March 2022 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import glob
import numpy
import matplotlib.pyplot as plt
import xrayutilities as xu

tt=numpy.arange(1,120,0.01)
powder=xu.simpack.Powder(xu.materials.Al,1)
pm=xu.simpack.PowderModel(powder)
ysim=pm.simulate(tt)
args=numpy.where(ysim>numpy.median(ysim)*1e2)
plt.plot(tt,ysim)
plt.scatter(tt[args],ysim[args],s=1,color='k',zorder=10)
plt.yscale('log')
plt.savefig('Al.pdf')

ttlist=numpy.concatenate((tt[args][0],tt[args][numpy.where(numpy.gradient(tt[args])>0.02)],tt[args][-1]),axis=None)

offset=-1
broadening=0.2

for i,value in enumerate(ttlist):
	if i%2==0:
		start=value
	else:
		end=value
		print('CUT['+str(int((i+1)/2))+']='+str((start+offset*numpy.sin(numpy.radians(start))-broadening).round(2))+':'+str((end+offset*numpy.sin(numpy.radians(end))+broadening).round(2)))
