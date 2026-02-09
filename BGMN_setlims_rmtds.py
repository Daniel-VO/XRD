"""
Created 09. Februar 2026 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob

limit=1+float(input('Grenzen Parameter A B C ALPHA BETA GAMMA, in Prozent (0=infty): '))/100
print(limit)

params=['A','B','C','ALPHA','BETA','GAMMA']

for f in glob.glob('*.str'):
	if 'amorphous' not in f:
		print(f)
		for l in open(f):
			for s in l.split(' '):
				for item in s.split('='):
					for p in params:
						if item==p and 'E=' not in s:
							startval=float(s.split('=')[-1].split('_')[0])
							if limit==1.0:
								news=s.replace(s.split('=')[-1],'%.6f'%startval)
							else:
								news=s.replace(s.split('=')[-1],'%.6f'%startval+'_'+'%.6f'%(startval/limit)+'^'+'%.6f'%(startval*limit))
							os.system('sed -i s/'+s+'/'+news+'/g '+f)
			if ' TDS=' in l:
				os.system('sed -i s/'+l.split(' ')[-1][:-1]+'//g '+f)
