import os
import glob

limit=1+float(input('Grenzen Parameter A B C ALPHA BETA GAMMA, in Prozent: '))/100

params=['A','B','C','ALPHA','BETA','GAMMA']

for f in glob.glob('*[!amorphous].str'):
	for l in open(f):
		for s in l.split(' '):
			for item in s.split('='):
				for p in params:
					if item==p and 'E=' not in s:
						startval=float(s.split('=')[-1].split('_')[0])
						news=s.replace(s.split('=')[-1],'%.6f'%startval+'_'+'%.6f'%(startval/limit)+'^'+'%.6f'%(startval*limit))
						os.system("sed -i 's/"+s+"/"+news+"/g' "+f)
