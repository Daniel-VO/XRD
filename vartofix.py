"""
Created 18. March 2022 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import numpy, glob, os
import matplotlib.pyplot as plt

files=glob.glob('*[!_corr].xy')

for i in files:
	print(i)
	zwotheta,yobs=numpy.genfromtxt(i,unpack=True)
	# ~ plt.clf()
	# ~ plt.plot(zwotheta,yobs/max(yobs))
	if numpy.median(yobs[-10:])>numpy.median(yobs[:int(len(yobs)/2)])/2:
		yobs/=numpy.sin(numpy.radians(zwotheta)/2)
		# ~ plt.plot(zwotheta,yobs/max(yobs))
		# ~ plt.show()
		export=numpy.array([zwotheta,yobs])
		numpy.savetxt(os.path.splitext(i)[0]+'_corr.xy',export.transpose(),newline='\n')
	else:
		print('Apparently fixed slit')
